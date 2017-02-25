###########load libraiy#############
library(e1071) #svm
library(class) #knn
library(pROC)  #roc auc
library(randomForest)  #random forest
library(nnet) #neural network
library(scatterplot3d) #plot
library(graphics) #plot


##########Data preparation##########
# Read in angle closure data
myData=read.csv("AngleClosure.csv",header=TRUE,na.strings=c("NA","."))
# Set-up response and predictor variables
# remove 13 predictor columns as required
myResponse=as.numeric(myData$ANGLE.CLOSURE=="YES")###
myPredictors=data.matrix(myData[,!(attributes(myData)$names %in% 
                                     c("EYE","GENDER","ETHNIC"))]);
# Remove rows with any missingness
myLogical=apply(cbind(myResponse,myPredictors),1,function(xx){
  return(!any(is.na(xx)))
})
myResponse=myResponse[myLogical]
myPredictors=data.matrix(myData[,!(attributes(myData)$names %in% 
                                     c("EYE","GENDER","ETHNIC","ANGLE.CLOSURE","HGT","WT","ASPH","ACYL","SE","AXL","CACD","AGE","CCT.OD","PCCURV_mm"))]);
myPredictors=myPredictors[myLogical,]



############probelm 4 stacking###########################
#initialization
nIter=10
nFolds=10
jj=1
testLength=ceiling(length(myResponse)/nFolds)
#testLength=137
u_matrix=matrix(NA,testLength*nIter,5)
y_matrix=rep(NA,testLength*nIter)

count=1
for (i in 1:nIter){
  
  myIndices=sample(length(myResponse))
  myResponseTraining=
    myResponse[myIndices[-c(((jj-1)*ceiling(length(myResponse)/nFolds)+1):
                              min(c(length(myResponse),jj*ceiling(length(myResponse)/nFolds))))]]
  myPredictorsTraining=
    myPredictors[myIndices[-c(((jj-1)*ceiling(length(myResponse)/nFolds)+1):
                                min(c(length(myResponse),jj*ceiling(length(myResponse)/nFolds))))],]
  myResponseTesting=
    myResponse[myIndices[c(((jj-1)*ceiling(length(myResponse)/nFolds)+1):
                             min(c(length(myResponse),jj*ceiling(length(myResponse)/nFolds))))]]
  myPredictorsTesting=
    myPredictors[myIndices[c(((jj-1)*ceiling(length(myResponse)/nFolds)+1):
                               min(c(length(myResponse),jj*ceiling(length(myResponse)/nFolds))))],]
  
  
  
  #svm
  #> svm_auc_avg[svm_best_index,]
  #[1] 50.0000000  0.0010000  0.9587051
  #[cost,gamma,auc]
  gammaValue=0.001
  costValue=50
  myX=myPredictorsTraining
  svm_model=svm(myResponseTraining~myX,type="C",kernel="radial",
                cost=costValue,gamma=gammaValue,probability=TRUE)
  myX=myPredictorsTesting
  svm_fit=predict(svm_model,myX, probability=TRUE)
  svm_u=attr(svm_fit,"probabilities")[,1]
  
  
  #knn
  #> knn_auc_avg[knn_best_index,]
  #[1] 110.0000000   0.8823281
  #[k,auc]
  kValue=110
  knn_fit=knn(myPredictorsTraining,myPredictorsTesting,myResponseTraining,k=kValue,prob=TRUE)
  knn_u=1-attr(knn_fit,"prob")
  
  
  #random forest
  #> rf_auc_avg[rf_best_index,]
  #[1] 2.000000 0.952731
  #[mtry,auc]
  mtryValue=2
  rf_model=randomForest(y=myResponseTraining,x=myPredictorsTraining,n.trees=5000,mtry=mtryValue)
  rf_u=as.matrix(predict(rf_model,myPredictorsTesting))
  
  
  #logistic regression
  lr_model=step(glm(myResponseTraining~1,data=data.frame(myPredictorsTraining),family="binomial"),
                scope=myResponseTraining~AOD750+TISA750+IT750+IT2000+ITCM+IAREA+ICURV+ACW_mm+ACA+ACV+LENSVAULT,
                trace=0)
  lr_u=predict(lr_model,newdata=data.frame(myPredictorsTesting),type="response")
  
  
  
  #neural network
  #> nueral_auc_avg[nueral_best_index,]
  #[1]  0.1000000 11.0000000  0.957348
  #[decay,size,auc]
  decayValue=0.1
  sizeValue=11
  myX=myPredictorsTraining
  nueral_model=nnet(myResponseTraining~myX,decay=decayValue,
                    size=sizeValue,trace=FALSE)
  myX=myPredictorsTesting
  nueral_u=predict(nueral_model,myX, probability=TRUE)
  
  
  #store u and y
  start=(count-1)*testLength+1
  end=count*testLength
  u_matrix[start:end,]=cbind(svm_u,knn_u,rf_u,lr_u,nueral_u)
  y_matrix[start:end]=myResponseTesting
  count=count+1
}



#stacked ensemble model should be based on the un-constrained (least squares) solution
unconstrained_b=solve((t(u_matrix)%*%u_matrix))%*%t(u_matrix)%*%y_matrix
unconstrained_y=u_matrix%*%unconstrained_b
unconstrained_roc=roc(y_matrix,unconstrained_y[,1])
unconstrained_auc=auc(unconstrained_roc)
#> unconstrained_auc
#Area under the curve: 0.9663
#> unconstrained_b
#[1,] -0.009322399
#[2,]  0.004239040
#[3,]  0.379356527
#[4,]  0.843672940
#[5,] -0.201107262


#stacked ensemble model subject to sum(w)=1, w>=0
library(quadprog)
#set constraint matrix (Amat)
constraint_matrix=matrix(0,5,6)
constraint_matrix[,1]=1
for (i in seq(5)) {
  constraint_matrix[i,i+1]=1
}
#vext holding values of b0
beta0_vector=c(1,0,0,0,0,0)
constrained_b=solve.QP(t(u_matrix)%*%u_matrix,
         t(y_matrix)%*%u_matrix,
         constraint_matrix,beta0_vector,meq=1)$solution
constrained_y=u_matrix%*%constrained_b
constrained_roc=roc(y_matrix,constrained_y[,1])
constrained_auc=auc(constrained_roc)
#> constrained_b
#[1] -6.076985e-19 -1.169076e-17  3.183279e-01  6.816721e-01  0.000000e+00
#> constrained_auc
#Area under the curve: 0.9674


###################################################################
