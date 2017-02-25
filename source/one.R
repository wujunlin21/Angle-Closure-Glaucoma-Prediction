###########load libraiy#############
library(e1071) #svm
library(class) #knn
library(pROC)  #roc auc
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




###############Question 2, 3, 6 ##################
##Develop Prediction Models, Tuning parameter, and visualize AUC vs. tuning parameter values##


##################################################
# Set-up 10-fold cross-validation
#set.seed(30126)##########
nFolds=10
nIter=10

#########SVM tuning cost and gamma (kernel=radial)######
# Set-up 10-fold cross-validation
nFolds=10
nIter=10
# Initialize vectors to store contributions to loss
costList=c(0.01,0.05,0.1,0.5,1,5,10,50)
gammaList=c(0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1)
svm_auc_raw=matrix(NA,length(costList)*length(gammaList),2+nIter)
count=1
for (c in costList) {
  for (g in gammaList) {
    svm_auc_raw[count,1]=c
    svm_auc_raw[count,2]=g
    count=count+1
  }
}
#10 fold cost validation with 10 iteration
for (ii in seq(nIter)) {
  myIndices=sample(length(myResponse))
  count=1
  for (cc in seq(length(costList))) {
  for (gg in seq(length(gammaList))) {
    fold_auv=rep(0,nFolds)
    for(jj in 1:nFolds){
  
      # Generate training and testing responses and predictors for each fold
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

     #train svm model on training data
      myX=myPredictorsTraining
      svm_model=svm(myResponseTraining~myX,type="C",kernel="radial",
                     cost=costList[cc],gamma=gammaList[gg],probability=TRUE)
      #fit svm model into testing data
      myX=myPredictorsTesting
      svm_fit=predict(svm_model,myX, probability=TRUE)
      svm_pro=attr(svm_fit,"probabilities")[,1]
      svm_roc=roc(myResponseTesting,svm_pro)
      svm_auc=auc(svm_roc)
      #calculate average auc for 10 fold
      fold_auv[jj]=svm_auc
    }
    #store average auc for each cost and gamma
    svm_auc_raw[count,ii+2]=mean(fold_auv)
    count=count+1
  }
  }
} 
#calculate average auc over 10 iterations
svm_auc_avg=matrix(NA,dim(svm_auc_raw)[1],3) 
svm_auc_avg[,1:2]=svm_auc_raw[,1:2]
for (count in seq(dim(svm_auc_raw)[1])){
  svm_auc_avg[count,3]=mean(svm_auc_raw[count,3:nIter+2])##
}
#find best parameters
svm_best_index=which(svm_auc_avg[,3]==max(svm_auc_avg[,3]))
#> svm_best_index
#[1] 58
svm_auc_avg[svm_best_index,]
#> svm_auc_avg[svm_best_index,]
#[1] 50.0000000  0.0010000  0.9587051
#[cost,gamma,auc]

#plot auc vs. (cost,gamma)
scatterplot3d(svm_auc_avg[,1],svm_auc_avg[,2],svm_auc_avg[,3]
              ,pch=16, highlight.3d=TRUE,
              type="h", main="AUC for svm",xlab="cost",ylab="gamma",zlab="AUC") 
filled.contour(costList,gammaList,
               matrix(svm_auc_avg[,3],
                      length(costList),
                      length(gammaList))
                       ,xlab="cost",ylab="gamma",main="SVM AUC")



######################################################################
