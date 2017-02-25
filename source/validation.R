###########load libraiy#############
library(e1071) #svm
library(class) #knn
library(pROC)  #roc auc
library(randomForest)  #random forest
library(nnet) #neural network
library(scatterplot3d) #plot
library(graphics) #plot


#######Training Data Preparation#######
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



##########Validation Data preparation##########
# Read in angle closure data
caseData=read.csv("AngleClosure_ValidationCases.csv",header=TRUE)
controlData=read.csv("AngleClosure_ValidationControls.csv",header=TRUE)
#prepare case predictors
casePredictors=c()
for (i in c(7,9,11:15)) {
  temp=apply(cbind(caseData[,i],caseData[,i+12]),1,function(xx){
  if(!is.na(xx[2])){ return(xx[2])}
  else{return(xx[1])}
  }) 
  casePredictors=cbind(casePredictors,temp)
  colnames(casePredictors)[dim(casePredictors)[2]]=substring(colnames(caseData)[i],2)
}
casePredictors=cbind(casePredictors,caseData[c(30:32,36)])
myLogical=apply(casePredictors,1,function(xx){
  return(!any(is.na(xx)))
})
casePredictors=casePredictors[myLogical,]
#prepare control data
controlPredictors=c()
for (i in c(6,8,10:14)) {
  temp=apply(cbind(controlData[,i],controlData[,i+12]),1,function(xx){
    if(!is.na(xx[2])){ return(xx[2])}
    else{return(xx[1])}
  }) 
  controlPredictors=cbind(controlPredictors,temp)
  colnames(controlPredictors)[dim(controlPredictors)[2]]=substring(colnames(controlData)[i],2)
}
controlPredictors=cbind(controlPredictors,controlData[c(29:31,35)])
colnames(controlPredictors)[8]="ACWmm"
colnames(controlPredictors)[7]="ICURV"
myLogical=apply(controlPredictors,1,function(xx){
  return(!any(is.na(xx)))
})
controlPredictors=controlPredictors[myLogical,]
#response data preparation
caseValidation=cbind(rep(1,dim(casePredictors)[1]),casePredictors)
colnames(caseValidation)[1]="ANGLE.CLOSURE"
controlValidation=cbind(rep(0,dim(controlPredictors)[1]),controlPredictors)
colnames(controlValidation)[1]="ANGLE.CLOSURE"
#final validation data
validationData=validation=rbind(caseValidation,controlValidation)
#> dim(validationData)
#[1] 400  12
validationX=validationY=validationData[,2:12]
validationY=validationData[,1]


#######################################################

######Validation SVM #########
gammaValue=0.001
costValue=50

dev.new(width=1.2*3,height=1.2*3)
par(mai=c(0.5,0.5,0.3,0.05),cex=0.8)
plot(roc(as.numeric(Yvalid),predict.svm.u),main="SVM",print.auc=TRUE)


myX=myPredictors
svm_model=svm(myResponse~myX,type="C",kernel="radial",
              cost=costValue,gamma=gammaValue,probability=TRUE)
myX=as.matrix(validationX)
svm_fit=predict(svm_model,myX, probability=TRUE)
svm_pro=attr(svm_fit,"probabilities")[,1]
svm_roc=roc(as.numeric(validationY),svm_pro)
svm_auc=auc(svm_roc)
plot(svm_roc,main="SVM",print.auc=TRUE)
#########Area under the curve: 0.9491##########




##########Validation KNN #########
kValue=110
knn_fit=knn(myPredictors,validationX,myResponse,k=kValue,prob=TRUE)
knn_u=1-attr(knn_fit,"prob")
knn_roc=roc(as.numeric(validationY),knn_u)
knn_auc=auc(knn_roc)
plot(knn_roc,main="KNN",print.auc=TRUE)
########Area under the curve: 0.9405###########


#######Validation random forest########
mtryValue=2
colnames(myPredictors)[8]="ACWmm"
rf_model=randomForest(y=myResponse,x=myPredictors,n.trees=5000,mtry=mtryValue)
rf_u=as.matrix(predict(rf_model,as.matrix(validationX)))
rf_roc=roc(as.numeric(validationY),as.numeric(rf_u))
rf_auc=auc(rf_roc)
plot(rf_roc,main="Random Forest",print.auc=TRUE)
#####Area under the curve: 0.9584#####


######Validation logistic regression with AIC#######
lr_model=step(glm(myResponse~1,data=data.frame(myPredictors),family="binomial"),
              scope=myResponse~AOD750+TISA750+IT750+IT2000+ITCM+IAREA+ICURV+ACWmm+ACA+ACV+LENSVAULT,
              trace=0)
lr_u=predict(lr_model,newdata=data.frame(validationX),type="response")
lr_roc=roc(as.numeric(validationY),as.numeric(lr_u))
lr_auc=auc(lr_roc)
plot(lr_roc,main="Logistics Regression AIC",print.auc=TRUE)
#########Area under the curve: 0.9539###########


######Validation neural network##########
decayValue=0.1
sizeValue=11
myX=myPredictors
nueral_model=nnet(myResponse~myX,decay=decayValue,
                  size=sizeValue,trace=FALSE)
myX=as.matrix(validationX)
nueral_u=predict(nueral_model,myX, probability=TRUE)
nueral_roc=roc(as.numeric(validationY),as.numeric(nueral_u))
nueral_auc=auc(nueral_roc)
plot(nueral_roc,main="neural network",print.auc=TRUE)
#################Area under the curve: 0.9702###########



#########Validate Stacking Unconstrained######
u_matrix=cbind(svm_pro,knn_u,rf_u,lr_u,nueral_u)
#> unconstrained_b
#[1,] -0.009322399
#[2,]  0.004239040
#[3,]  0.379356527
#[4,]  0.843672940
#[5,] -0.201107262
unconstrained_b=c(-0.009322399,0.004239040,0.379356527,
                  0.843672940,-0.201107262)
unconstrained_y=u_matrix%*%unconstrained_b
unconstrained_roc=roc(validationY,unconstrained_y[,1])
unconstrained_auc=auc(unconstrained_roc)
plot(unconstrained_roc,main="unconstrained stacking",print.auc=TRUE)
######Area under the curve: 0.9541##############


#########Validate Stacking Constrained########
#> constrained_b
#[1] -6.076985e-19 -1.169076e-17  3.183279e-01  6.816721e-01  0.000000e+00
constrained_b=c(-6.076985e-19, -1.169076e-17, 3.183279e-01
                  ,6.816721e-01,  0.000000e+00)
constrained_y=u_matrix%*%constrained_b
constrained_roc=roc(validationY,constrained_y[,1])
constrained_auc=auc(constrained_roc)
plot(constrained_roc,main="constrained stacking",print.auc=TRUE)
######Area under the curve: 0.9584#############

