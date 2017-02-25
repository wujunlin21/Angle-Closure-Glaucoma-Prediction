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


#########KNN tuning k######
# Set-up 10-fold cross-validation
nFolds=10
nIter=25
# Initialize vectors to store contributions to loss
nNNList=seq(10,120,10)
knn_auc_raw=matrix(NA,length(nNNList),1+nIter)
knn_auc_raw[,1]=nNNList
#10 fold cost validation with 10 iteration
for (ii in seq(nIter)) {
  myIndices=sample(length(myResponse))
  count=1
  for (nNN in seq(length(nNNList))) {
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

      #train and fit kNN model
      knn_fit=knn(myPredictorsTraining,myPredictorsTesting,myResponseTraining,k=nNNList[nNN],prob=TRUE)
      knn_pro=attr(knn_fit,"prob")
      knn_roc=roc(myResponseTesting-1,knn_pro)
      knn_auc=auc(knn_roc)
      #calculate average auc for 10 fold
      fold_auv[jj]=knn_auc
    }
    #store average auc for each cost and gamma
    knn_auc_raw[count,ii+1]=mean(fold_auv)
    count=count+1
  }
} 
#calculate average auc over 10 iterations
knn_auc_avg=matrix(NA,dim(knn_auc_raw)[1],2) 
knn_auc_avg[,1]=knn_auc_raw[,1]
for (count in seq(dim(knn_auc_raw)[1])){
  knn_auc_avg[count,2]=mean(knn_auc_raw[count,2:dim(knn_auc_raw)[2]])##
}
#find best parameters
knn_best_index=which(knn_auc_avg[,2]==max(knn_auc_avg[,2]))
#> knn_best_index
#[1] 11
knn_auc_avg[knn_best_index,]
#> knn_auc_avg[knn_best_index,]
#[1] 110.0000000   0.8823281
#[k,auc]

#plot auc vs. num of nearest number
plot(knn_auc_avg[,1],knn_auc_avg[,2],type="l",
     xlab="num of nearest number",ylab="AUC",main="kNN")



######################################################################
