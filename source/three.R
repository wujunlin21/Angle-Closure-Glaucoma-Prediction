###########load libraiy#############
library(e1071) #svm
library(class) #knn
library(pROC)  #roc auc
library(scatterplot3d) #plot
library(graphics) #plot
library(randomForest)  #random forest


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


#########random forest tuning mtry#####################
#########(Number of variables randomly sampled as candidates at each split) ######
# Set-up 10-fold cross-validation
nFolds=10
nIter=10
# Initialize vectors to store contributions to loss
mtryList=seq(2,11)
rf_auc_raw=matrix(NA,length(mtryList),1+nIter)
rf_auc_raw[,1]=mtryList
#10 fold cost validation with 10 iteration
for (ii in seq(nIter)) {
  myIndices=sample(length(myResponse))
  count=1
  for (mm in seq(length(mtryList))) {
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

      #train and fit rf model
      rf_model=randomForest(y=myResponseTraining,x=myPredictorsTraining,n.trees=5000,mtry=mtryList[mm])
      rf_fit=predict(rf_model,myPredictorsTesting)
      rf_roc=roc(myResponseTesting,rf_fit)
      rf_auc=auc(rf_roc)
      #calculate average auc for 10 fold
      fold_auv[jj]=rf_auc
    }
    #store average auc for each cost and gamma
    rf_auc_raw[count,ii+1]=mean(fold_auv)
    count=count+1
  }
} 
#calculate average auc over 10 iterations
rf_auc_avg=matrix(NA,dim(rf_auc_raw)[1],2) 
rf_auc_avg[,1]=rf_auc_raw[,1]
for (count in seq(dim(rf_auc_raw)[1])){
  rf_auc_avg[count,2]=mean(rf_auc_raw[count,2:dim(rf_auc_raw)[2]])##
}
#find best parameters
rf_best_index=which(rf_auc_avg[,2]==max(rf_auc_avg[,2]))
#> rf_best_index
#[1] 1
rf_auc_avg[rf_best_index,]
#> rf_auc_avg[rf_best_index,]
#[1] 2.000000 0.952731
#[mtry,auc]

#plot auc vs. mtry
plot(rf_auc_avg[,1],rf_auc_avg[,2],type="l",
     xlab="mtry",ylab="AUC",main="Random Forest")



######################################################################

#########Logistic regression with AIC##########
# Set-up 10-fold cross-validation
nFolds=10
nIter=25
# Initialize vectors to store contributions to loss
lr_auc_raw=rep(NA,nIter)
#10 fold cost validation with 10 iteration
count=1
for (ii in seq(nIter)) {
  myIndices=sample(length(myResponse))
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
      
      #train and fit logistic regression model
      lr_model=step(glm(myResponseTraining~1,data=data.frame(myPredictorsTraining),family="binomial"),
                    scope=myResponseTraining~AOD750+TISA750+IT750+IT2000+ITCM+IAREA+ICURV+ACW_mm+ACA+ACV+LENSVAULT,
                    trace=0)
      #Step=3:  AIC=509.91
      lr_fit=predict(lr_model,newdata=data.frame(myPredictorsTesting))
      lr_roc=roc(myResponseTesting,lr_fit)
      lr_auc=auc(lr_roc)
      #calculate average auc for 10 fold
      fold_auv[jj]=lr_auc
  }
    #store average auc for each cost and gamma
  lr_auc_raw[count]=mean(fold_auv)
  count=count+1
} 
#calculate average auc over 10 iterations
lr_auc_avg=mean(lr_auc_raw)
#> lr_auc_avg
#[1] 0.958219



#####################################################################


