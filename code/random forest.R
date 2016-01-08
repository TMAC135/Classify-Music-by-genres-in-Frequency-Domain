library(party)
library(randomForest)
ml = read.csv("~/Desktop/5521.csv",header=FALSE)

############################################################
# 10 labels case
# set up the true labels
label = c(rep(1,100),rep(2,100),rep(3,100),rep(4,100),rep(5,100),
  rep(6,100),rep(7,100),rep(8,100),rep(9,100),rep(10,100))
set.seed(2)

# choose the training validation group randomly
index=sample(1000,1000)
tr=index[1:800]
va=index[801:1000]
trla=label[tr]
vala=label[va]
X=ml
train=X[tr,]
valid=X[va,]

# random forest with ntree=500, variable selection and pick the optimal m
rf.m=tuneRF(y=as.factor(trla), x=train,ntree=500,doBest=TRUE,trace=FALSE,importance =TRUE)

# the predicted labels for validation group
pred=predict(rf.m,newdata=valid)
preds= as.numeric(pred)

error=0
for (i in 1:length(vala))
{
  if (preds[i]!=vala[i])
  {
    error=error+1
  }
}
# error rate with 10 labels
error/200

############################################################
# 4 labels case
# set up the true labels
label2 = c(rep(1,100),rep(2,100),rep(3,100),rep(4,100))
set.seed(7)

# choose the training validation group randomly
index2=sample(400,400)
tr2=index2[1:320]
va2=index2[321:400]
trla2=label2[tr2]
vala2=label2[va2]
X2=ml[c(101:200,401:500,601:700,801:900),]
train2=X2[tr2,]
valid2=X2[va2,]

# random forest with ntree=500, variable selection and pick the optimal m
rf.m2=tuneRF(y=as.factor(trla2), x=train2,ntree=500,doBest=TRUE,trace=FALSE,importance =TRUE)

# the predicted labels for validation group
pred2=predict(rf.m2,newdata=valid2)
preds2= as.numeric(pred2)

error2=0
for (i in 1:length(vala2))
{
  if (preds2[i]!=vala2[i])
  {
    error2=error2+1      
  }
}
# error rate with 10 labels
error2/80

temp=c(0,0)
error2=0
for (i in 1:length(vala2))
{
  if (preds2[i]!=vala2[i])
  {
    error2=error2+1
    temp[error2] = i
  }
}
