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




###############Question 2, 3, 6 ##################
##Develop Prediction Models, Tuning parameter, and visualize AUC vs. tuning parameter values##


##################################################


#########nueral network tuning dacay and size######
########decay: parameter for weight decay#######
########size: number of units in the hidden layer###
# Set-up 10-fold cross-validation
nFolds=10
nIter=10
# Initialize vectors to store contributions to loss
dacayList=c(0.00001,0.0001,0.001,0.01,0.1,1,10)
sizeList=seq(2,12)
nueral_auc_raw=matrix(NA,length(dacayList)*length(sizeList),2+nIter)
count=1
for (d in dacayList) {
  for (s in sizeList) {
    nueral_auc_raw[count,1]=d
    nueral_auc_raw[count,2]=s
    count=count+1
  }
}
#10 fold cost validation with 10 iteration
for (ii in seq(2,nIter)) {
  myIndices=sample(length(myResponse))
  count=1
  for (dd in seq(length(dacayList))) {
    for (ss in seq(length(sizeList))) {
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
        #train nueral network model on training data
        myX=myPredictorsTraining
        nueral_model=nnet(myResponseTraining~myX,decay=dacayList[dd],
                      size=sizeList[ss],trace=FALSE)
        #fit svm model into testing data
        myX=myPredictorsTesting
        nueral_fit=predict(nueral_model,myX, probability=TRUE)
        #nueral_pro=attr(nueral_fit,"probabilities")[,1]
        nueral_roc=roc(myResponseTesting,nueral_fit[,1])
        nueral_auc=auc(nueral_roc)
        #calculate average auc for 10 fold
        fold_auv[jj]=nueral_auc
      }
      #store average auc for each cost and gamma
      nueral_auc_raw[count,ii+2]=mean(fold_auv)
      count=count+1
    }
  }
} 
#calculate average auc over 10 iterations
nueral_auc_avg=matrix(NA,dim(nueral_auc_raw)[1],3) 
nueral_auc_avg[,1:2]=nueral_auc_raw[,1:2]
for (count in seq(dim(nueral_auc_raw)[1])){
  nueral_auc_avg[count,3]=mean(nueral_auc_raw[count,3:nIter+2])##
}
#find best parameters
nueral_best_index=which(nueral_auc_avg[,3]==max(nueral_auc_avg[,3]))
#> nueral_best_index
#[1] 54
nueral_auc_avg[nueral_best_index,]
#> nueral_auc_avg[nueral_best_index,]
#[1]  0.1000000 11.0000000  0.957348
#[decay,size,auc]

#plot auc vs. (cost,gamma)
scatterplot3d(nueral_auc_avg[,1],nueral_auc_avg[,2],nueral_auc_avg[,3]
              ,pch=16, highlight.3d=TRUE,
              type="h", main="AUC for nueral network",
              xlab="decay",ylab="size",zlab="AUC") 
filled.contour(dacayList,sizeList,
               matrix(nueral_auc_avg[,3],
                      length(dacayList),
                      length(sizeList))
               ,xlab="decay",ylab="size",main="Neural Network AUC Contour")



######################################################################