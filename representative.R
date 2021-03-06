############ Prereqs ############
args<-commandArgs(trailingOnly=TRUE)
iter<-args[1]

library(methods)
library(clusterGeneration)
library(randomForest)
library(e1071)
library(neuralnet)
library(caret)
library(class)
library(cvTools)
library(qvalue)
library(dplyr)
library(tidyr)

# Phenotype distribution:
phePriorTrain<-MCMCpack::rdirichlet(1,c(5,5,5))
phePriorTest<-MCMCpack::rdirichlet(1,c(5,5,5))

pheDistributionTrain<-c(rmultinom(1,60,prob=phePriorTrain))
pheDistributionTest<-c(rmultinom(1,60,prob=phePriorTest))

pheDistribution<-pheDistributionTrain+pheDistributionTest
pheDN<-cumsum(pheDistribution)

phe<-factor(c(rep("phe1",pheDistribution[1]),rep("phe2",pheDistribution[2]),
              rep("phe3",pheDistribution[3])))
mmPhe<-model.matrix(~phe-1,as.data.frame(phe))

# Train phe:
pheTrain<-factor(c(rep("phe1",pheDistributionTrain[1]),rep("phe2",pheDistributionTrain[2]),
              rep("phe3",pheDistributionTrain[3])))
mmPheTrain<-model.matrix(~pheTrain-1,as.data.frame(pheTrain))

# Test phe:
pheTest<-factor(c(rep("phe1",pheDistributionTest[1]),rep("phe2",pheDistributionTest[2]),
              rep("phe3",pheDistributionTest[3])))
mmPheTest<-model.matrix(~pheTest-1,as.data.frame(pheTest))

eta<-.01

# Pathways differentially abundant
perBlocksPhe2<-sample(2:6,1)
blockMeansPhe2<-rexp(perBlocksPhe2,rate=1/2)*sample(c(-1,1),perBlocksPhe2,replace=TRUE)
perBlocksPhe3<-sample(2:6,1)
blockMeansPhe3<-rexp(perBlocksPhe3,rate=1/2)*sample(c(-1,1),perBlocksPhe3,replace=TRUE)

#Baseline / Phe 1 data:
sigmas<-replicate(25,genPositiveDefMat(40,covMethod="c-vine",eta=eta)$Sigma,simplify=FALSE)
metabs<-lapply(sigmas,function(x) mvrnorm(n=120,mu=rep(0,40),Sigma=x))
metabs<-do.call(cbind,metabs)
colnames(metabs)<-paste("m",as.vector(t(outer(1:25,1:40,paste,sep="."))),sep=".")

#Phe 2 data:
whichBlocksPhe2<-sample(1:25,perBlocksPhe2)
permPhe2<-data.frame(start=40*(whichBlocksPhe2-1)+1)
permPhe2$end<-permPhe2$start+39
for(i in 1:perBlocksPhe2)
{
  metabs[(pheDN[1]+1):(pheDN[2]),permPhe2$start[i]:permPhe2$end[i]]<-
    mvrnorm(n=pheDistribution[2],mu=rep(blockMeansPhe2[i],40),Sigma=sigmas[[whichBlocksPhe2[i]]])
}

#Phe 3 data:
whichBlocksPhe3<-sample(1:25,perBlocksPhe3)
permPhe3<-data.frame(start=40*(whichBlocksPhe3-1)+1)
permPhe3$end<-permPhe3$start+39
for(i in 1:perBlocksPhe3)
{
  metabs[(pheDN[2]+1):(pheDN[3]),permPhe3$start[i]:permPhe3$end[i]]<-
    mvrnorm(n=pheDistribution[3],mu=rep(blockMeansPhe3[i],40),Sigma=sigmas[[whichBlocksPhe3[i]]])
}

# Add outlier clusters within replicate
outliers<-rpois(120,1)
for(i in 1:length(outliers))
{
  if(outliers[i]>0L)
  {
    outClusts<-sample(1:25,outliers[i])
    for(outClust in outClusts)
    {
      clustMeans<-mean(apply(metabs[,((outClust-1)*40+1):(outClust*40)],2,mean))
      clustVar<-mean(apply(metabs[,((outClust-1)*40+1):(outClust*40)],2,sd))
      # Add cluster level outlier:
      metabs[i,((outClust-1)*40+1):(outClust*40)]<-
        mvrnorm(n=1,mu=rep(clustMeans+2*sample(c(-1,1),1)*clustVar,40),Sigma=sigmas[[outClust]])
    } 
  }
}

# Add outlier not within clusters:
outliers<-rpois(120,2)
for(i in 1:length(outliers))
{
  if(outliers[i]>0L)
  {
    outs<-sample(1:1000,outliers[i])
    for(out in outs)
    {
      metabMean<-mean(metabs[,out])
      metabSD<-sd(metabs[,out])
      metabs[i,out]<-rnorm(n=1,mean=metabMean+sample(c(-1,1),1)*2*metabSD)
    }
  }
}

# Introduce non-normal error distributions:
for(i in 1:25)
{
  nonNormInd<-as.logical(rbinom(1,size=1,prob=.5))
  if(nonNormInd)
  {
    alpha<-rexp(1,rate=1/4)
    kappa<-runif(1,-1,1)
    fun1<-function(quants,alpha,kappa)
    {
      return(alpha/kappa*(1-exp(kappa*qnorm(quants-1e-3))))
    }
    for(j in 1:40)
    {
      kk<-(i-1)*40+j
      nonNormReplace<-fun1(ecdf(metabs[,kk])(metabs[,kk]),alpha,kappa)
      metabs[,kk]<-nonNormReplace
    }
  }
}

# Not detected missing values:
metabs[metabs<quantile(metabs,rbeta(1,shape1=1,shape2=19))]<-NA
minImp<-function(x)
{
  minX<-min(x[!is.na(x)])
  x[is.na(x)]<-minX
  return(x)
}
metabs<-apply(metabs,2,FUN=minImp)

rownames(metabs)<-c(paste(rep("phe1",pheDistribution[1]),1:(pheDistribution[1]),sep="."),
  paste(rep("phe2",pheDistribution[2]),1:(pheDistribution[2]),sep="."),
  paste(rep("phe3",pheDistribution[3]),1:(pheDistribution[3]),sep="."))

trainRows<-c(paste(rep("phe1",pheDistributionTrain[1]),1:(pheDistributionTrain[1]),sep="."),
  paste(rep("phe2",pheDistributionTrain[2]),1:(pheDistributionTrain[2]),sep="."),
  paste(rep("phe3",pheDistributionTrain[3]),1:(pheDistributionTrain[3]),sep="."))

train<-metabs[rownames(metabs) %in% trainRows,]
test<-metabs[!rownames(metabs) %in% trainRows,]

# Significance filtering
varNames<-colnames(train)
pFun<-function(x)
{
  return(na.omit(
    c(pairwise.wilcox.test(x=train[,x],g=pheTrain,p.adjust.method="none")$p.value)))
}
gr<-do.call("rbind",lapply(varNames,pFun))
gr<-data.frame(sig=apply(gr,1,function(x) x[1]<.015 | x[2]<.015 | x[3]<.015))
rownames(gr)<-gr$metab<-varNames
gr<-gr %>% filter(sig)
trainSig<-train[,colnames(train)%in%gr$metab]
testSig<-test[,colnames(test)%in%gr$metab]

cvF<-cvFolds(n=nrow(train),K=5)

############ Naive Bayes ############
NB<-naiveBayes(train,pheTrain)
NBSig<-naiveBayes(trainSig,pheTrain)

############ k-NN ############
# Full:
knnP<-data.frame(nn=seq(1:20),CELoss=NA)
for(j in 1:nrow(knnP))
{
  CELoss<-c()
  for(i in 1:5)
  {
    KNN<-knn(train=train[cvF$subsets[cvF$which!=i],],test=train[cvF$subsets[cvF$which==i],],
             cl=pheTrain[cvF$subsets[cvF$which!=i]],k=knnP$nn[j],prob=TRUE)
    probs<-attr(KNN,"prob")
    probs<-log2(probs*model.matrix(~KNN-1,data.frame(phe=KNN)))
    probs[is.infinite(probs)]<-log2(1e-64)
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPheTrain[cvF$subsets[cvF$which==i],]*probs))
  }
  knnP$CELoss[j]<-mean(CELoss)
}
knnNN<-knnP$nn[which.min(knnP$CELoss)]
KNN<-knn(train,test,cl=pheTrain,k=knnNN,prob=TRUE)

# Filtered
for(j in 1:nrow(knnP))
{
  CELoss<-c()
  for(i in 1:5)
  {
    KNNSig<-knn(train=trainSig[cvF$subsets[cvF$which!=i],],test=trainSig[cvF$subsets[cvF$which==i],],
                cl=pheTrain[cvF$subsets[cvF$which!=i]],k=knnP$nn[j],prob=TRUE)
    probs<-attr(KNNSig,"prob")
    probs<-log2(probs*model.matrix(~KNNSig-1,data.frame(phe=KNNSig)))
    probs[is.infinite(probs)]<-log2(1e-64)
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPheTrain[cvF$subsets[cvF$which==i],]*probs))
  }
  knnP$CELoss[j]<-mean(CELoss)
}
knnNN<-knnP$nn[which.min(knnP$CELoss)]
KNNSig<-knn(train,test,cl=pheTrain,k=knnNN,prob=TRUE)

############ PLS-DA ############
# Full
plsdaP<-data.frame(nComp=seq(1:25),CELoss=NA)
for(j in 1:nrow(plsdaP))
{
  CELoss<-c()
  for(i in 1:5)
  {
    Plsda<-plsda(train[cvF$subsets[cvF$which!=i],],pheTrain[cvF$subsets[cvF$which!=i]],
                 ncomp=plsdaP$nComp[j])
    probs<-log2(predict(Plsda,train[cvF$subsets[cvF$which==i],],type="prob")[,,1])
    probs[is.infinite(probs)]<-log2(1e-64)
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPheTrain[cvF$subsets[cvF$which==i],]*probs))
  }
  plsdaP$CELoss[j]<-mean(CELoss)
}
plsdaNComp<-plsdaP$nComp[which.min(plsdaP$CELoss)]
Plsda<-plsda(train,pheTrain,ncomp=plsdaNComp)
predPlsda<-predict(Plsda,newdata=test,type="prob")

# Filtered
plsdaP<-data.frame(nComp=seq(1:25),CELoss=NA)
for(j in 1:nrow(plsdaP))
{
  CELoss<-c()
  for(i in 1:5)
  {
    PlsdaSig<-plsda(trainSig[cvF$subsets[cvF$which!=i],],pheTrain[cvF$subsets[cvF$which!=i]],
                    ncomp=plsdaP$nComp[j])
    probs<-log2(predict(PlsdaSig,trainSig[cvF$subsets[cvF$which==i],],type="prob")[,,1])
    probs[is.infinite(probs)]<-log2(1e-64)
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPheTrain[cvF$subsets[cvF$which==i],]*probs))
  }
  plsdaP$CELoss[j]<-mean(CELoss)
}
plsdaNComp<-plsdaP$nComp[which.min(plsdaP$CELoss)]
PlsdaSig<-plsda(trainSig,pheTrain,ncomp=plsdaNComp)
predPlsdaSig<-predict(PlsdaSig,newdata=testSig,type="prob")

############ SPLS-DA ############
# Full:
splsdaP<-expand.grid(nComp=seq(1:15),eta=seq(.1,.9,.1),CELoss=NA)
for(j in 1:nrow(splsdaP))
{
  CELoss<-c()
  for(i in 1:5)
  {
    Splsda<-caret:::splsda(train[cvF$subsets[cvF$which!=i],],pheTrain[cvF$subsets[cvF$which!=i]],
                           eta=splsdaP$eta[j],K=splsdaP$nComp[j])
    probs<-log2(predict(Splsda,train[cvF$subsets[cvF$which==i],],type="prob"))
    probs[is.infinite(probs)]<-log2(1e-64)
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPheTrain[cvF$subsets[cvF$which==i],]*probs))
  }
  splsdaP$CELoss[j]<-mean(CELoss)
}
selEta<-splsdaP$eta[which.min(splsdaP$CELoss)]
selNComp<-splsdaP$nComp[which.min(splsdaP$CELoss)]
Splsda<-splsda(train,pheTrain,eta=selEta,K=selNComp)

# Filtered:
for(j in 1:nrow(splsdaP))
{
  CELoss<-c()
  for(i in 1:5)
  {
    SplsdaSig<-caret:::splsda(trainSig[cvF$subsets[cvF$which!=i],],pheTrain[cvF$subsets[cvF$which!=i]],
                              eta=splsdaP$eta[j],K=splsdaP$nComp[j])
    probs<-log2(predict(SplsdaSig,trainSig[cvF$subsets[cvF$which==i],],type="prob"))
    probs[is.infinite(probs)]<-log2(1e-64)
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPheTrain[cvF$subsets[cvF$which==i],]*probs))
  }
  splsdaP$CELoss[j]<-mean(CELoss)
}
selEta<-splsdaP$eta[which.min(splsdaP$CELoss)]
selNComp<-splsdaP$nComp[which.min(splsdaP$CELoss)]
SplsdaSig<-splsda(trainSig,pheTrain,eta=selEta,K=selNComp)

############ Random Forest ############
# Full:
rFmtry<-data.frame(mtry=seq(5,ncol(train),length=25),CELoss=NA)
rFmtry$mtry<-round(rFmtry$mtry)
for(j in 1:nrow(rFmtry))
{
  CELoss<-c()
  for(i in 1:5)
  {
    rF<-randomForest(train[cvF$subsets[cvF$which!=i],],pheTrain[cvF$subsets[cvF$which!=i]],
                     mtry=rFmtry$mtry[j])
    probs<-log2(predict(rF,train[cvF$subsets[cvF$which==i],],type="prob"))
    probs[is.infinite(probs)]<-log2(1e-64)
    probs<-probs[,levels(pheTrain)]
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPheTrain[cvF$subsets[cvF$which==i],]*probs))
  }
  rFmtry$CELoss[j]<-mean(CELoss)
}
lSmooth<-ksmooth(x=rFmtry$mtry,y=rFmtry$CELoss,bandwidth=150)
selMtry<-round(lSmooth$x[which.min(lSmooth$y)])
rF<-randomForest(train,pheTrain,mtry=selMtry)

# Filtered:
rFmtry<-data.frame(mtry=round(seq(2,ncol(trainSig),length=25)),CELoss=NA)
for(j in 1:nrow(rFmtry))
{
  CELoss<-c()
  for(i in 1:5)
  {
    rFSig<-randomForest(trainSig[cvF$subsets[cvF$which!=i],],pheTrain[cvF$subsets[cvF$which!=i]],
                        mtry=rFmtry$mtry[j])
    probs<-log2(predict(rFSig,trainSig[cvF$subsets[cvF$which==i],],type="prob"))
    probs[is.infinite(probs)]<-log2(1e-64)
    probs<-probs[,levels(pheTrain)]
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPheTrain[cvF$subsets[cvF$which==i],]*probs))
  }
  rFmtry$CELoss[j]<-mean(CELoss)
}
lSmooth<-ksmooth(x=rFmtry$mtry,y=rFmtry$CELoss,bandwidth=10)
selMtry<-round(lSmooth$x[which.min(lSmooth$y)])
rFSig<-randomForest(trainSig,pheTrain,mtry=selMtry)

############ SVM ############
# Full:
svmP<-data.frame(gamma=10**(seq(-5,-1,length=1000)),CELoss=NA)
for(j in 1:nrow(svmP))
{
  CELoss<-c()
  for(i in 1:5)
  {
    SVM<-svm(train[cvF$subsets[cvF$which!=i],],pheTrain[cvF$subsets[cvF$which!=i]],
             gamma=svmP$gamma[j],kernel="radial",probability=TRUE)
    probs<-log2(attr(predict(SVM,train[cvF$subsets[cvF$which==i],],probability=TRUE),"probabilities"))
    probs[is.infinite(probs)]<-log2(1e-64)
    probs<-probs[,levels(pheTrain)]
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPheTrain[cvF$subsets[cvF$which==i],]*probs))
  }
  svmP$CELoss[j]<-mean(CELoss)
  print(j)
}
#plot(log10(svmP$gamma),svmP$CELoss)
#points(ksmooth(x=log10(svmP$gamma),y=svmP$CELoss,bandwidth=1/25),type="l",col="red")
lSmooth<-ksmooth(x=log10(svmP$gamma),y=svmP$CELoss,bandwidth=1/25)
selSVMP<-10**lSmooth$x[which.min(lSmooth$y)]
SVM<-svm(train,pheTrain,gamma=selSVMP,kernel="radial",probability=TRUE)

# Filtered:
svmP<-data.frame(gamma=10**(seq(-4,0,length=1000)),CELoss=NA)
for(j in 1:nrow(svmP))
{
  CELoss<-c()
  for(i in 1:5)
  {
    SVMSig<-svm(trainSig[cvF$subsets[cvF$which!=i],],pheTrain[cvF$subsets[cvF$which!=i]],
                gamma=svmP$gamma[j],kernel="radial",probability=TRUE)
    probs<-log2(attr(predict(SVMSig,trainSig[cvF$subsets[cvF$which==i],],probability=TRUE),"probabilities"))
    probs[is.infinite(probs)]<-log2(1e-64)
    probs<-probs[,levels(pheTrain)]
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPheTrain[cvF$subsets[cvF$which==i],]*probs))
  }
  svmP$CELoss[j]<-mean(CELoss)
  print(j)
}
#plot(log10(svmP$gamma),svmP$CELoss)
#points(ksmooth(x=log10(svmP$gamma),y=svmP$CELoss,bandwidth=1/25),type="l",col="red")
lSmooth<-ksmooth(x=log10(svmP$gamma),y=svmP$CELoss,bandwidth=1/25)
selSVMP<-10**lSmooth$x[which.min(lSmooth$y)]
SVMSig<-svm(trainSig,pheTrain,gamma=selSVMP,kernel="radial",probability=TRUE)

############ Neural Network #############
probNorm<-function(x) exp(x)/sum(exp(x))

DfTrain<-as.data.frame(cbind(mmPheTrain,train))
DfTest<-as.data.frame(cbind(mmPheTest,test))
DfTrainSig<-as.data.frame(cbind(mmPheTrain,trainSig))
DfTestSig<-as.data.frame(cbind(mmPheTest,testSig))

# Full 
NNP<-expand.grid(layers=1:2,nodes=seq(15,100,5),CELoss=NA)
NNP$nodes<-round(NNP$nodes/NNP$layers)
for(j in 1:nrow(NNP))
{
  CELoss<-c()
  for(i in 1:5)
  {
    cat("Full i: ",i," structure:",rep(NNP$nodes[j],times=NNP$layers[j]),"\n")
    tempDfTrain<-as.data.frame(cbind(mmPheTrain[cvF$subsets[cvF$which!=i],],
                                     train[cvF$subsets[cvF$which!=i],]))
    tempDfTest<-as.data.frame(cbind(mmPheTrain[cvF$subsets[cvF$which==i],],
                                    train[cvF$subsets[cvF$which==i],]))
    tempForm<-as.formula(paste("pheTrainphe1+pheTrainphe2+pheTrainphe3",
                               paste(names(tempDfTrain)[!grepl("phe",names(tempDfTrain))],collapse="+"),sep="~"))
    
    attempt<-0
    probs<-NULL
    while(is.null(probs)&&attempt<=3)
    {
      attempt<-attempt+1
      try(
        {
          nnet<-neuralnet(tempForm,data=tempDfTrain,act.fct="logistic",err.fct="ce",
                          linear.output=FALSE,hidden=rep(NNP$nodes[j],times=NNP$layers[j]),
                          rep=1,threshold=.00001,likelihood=TRUE)
          nnetRes<-nnet$result.matrix
          nnetResMin<-which.min(nnetRes["aic",])
          probs<-neuralnet::compute(nnet,covariate=tempDfTest[,4:ncol(tempDfTest)],
                                    rep=nnetResMin)$net.result
        }
      )
    }
    probs<-log2(t(apply(probs,1,probNorm)))
    probs[is.infinite(probs)]<-log2(1e-64)
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPheTrain[cvF$subsets[cvF$which==i],]*probs))
  }
  NNP$CELoss[j]<-mean(CELoss)
}
NNP$nodes<-NNP$nodes*NNP$layers
#plot(NNP$nodes[NNP$layers==1],NNP$CELoss[NNP$layers==1],type="p",ylim=c(0,1.6))
#points(NNP$nodes[NNP$layers==2],NNP$CELoss[NNP$layers==2],type="p",col="blue")
lSmoothDf<-c()
for(i in 1:2)
{
  lSmooth<-ksmooth(x=NNP$nodes[NNP$layers==i],y=NNP$CELoss[NNP$layers==i],bandwidth=20)
  #points(lSmooth$x,lSmooth$y,type="l")
  lSmoothDf<-rbind(lSmoothDf,data.frame(layers=i,
                                        nodes=round(lSmooth$x[which.min(lSmooth$y)]),CELoss=min(lSmooth$y)))
}
NNPBest<-lSmoothDf[which.min(lSmoothDf$CELoss),]
nnet<-neuralnet(tempForm,data=DfTrain,act.fct="logistic",err.fct="ce",
                linear.output=FALSE,hidden=rep(round(NNPBest$nodes[1]/NNPBest$layers[1]),times=NNPBest$layers[1]),
                rep=1,threshold=.00001)

# Filtered:
NNP<-expand.grid(layers=1:2,nodes=seq(15,100,5),CELoss=NA)
NNP$nodes<-round(NNP$nodes/NNP$layers)
for(j in 1:nrow(NNP))
{
  CELoss<-c()
  for(i in 1:5)
  {
    cat("Filtered i: ",i," structure:",rep(NNP$nodes[j],times=NNP$layers[j]),"\n")
    tempDfTrain<-as.data.frame(cbind(mmPheTrain[cvF$subsets[cvF$which!=i],],
                                     trainSig[cvF$subsets[cvF$which!=i],]))
    tempDfTest<-as.data.frame(cbind(mmPheTrain[cvF$subsets[cvF$which==i],],
                                    trainSig[cvF$subsets[cvF$which==i],]))
    tempForm<-as.formula(paste("pheTrainphe1+pheTrainphe2+pheTrainphe3",
                               paste(names(tempDfTrain)[!grepl("phe",names(tempDfTrain))],collapse="+"),sep="~"))
    
    attempt<-0
    probs<-NULL
    while(is.null(probs)&&attempt<=3)
    {
      attempt<-attempt+1
      try(
        {
          nnetSig<-neuralnet(tempForm,data=tempDfTrain,act.fct="logistic",err.fct="ce",
                             linear.output=FALSE,hidden=rep(NNP$nodes[j],times=NNP$layers[j]),
                             rep=1,threshold=.00001,likelihood=TRUE)
          nnetSigRes<-nnetSig$result.matrix
          nnetSigResMin<-which.min(nnetSigRes["aic",])
          probs<-neuralnet::compute(nnetSig,covariate=tempDfTest[,4:ncol(tempDfTest)],
                                    rep=nnetSigResMin)$net.result
        }
      )
    }
    probs<-log2(t(apply(probs,1,probNorm)))
    probs[is.infinite(probs)]<-log2(1e-64)
    CELoss<-c(CELoss,1/length(cvF$subsets[cvF$which==i])*
                sum(-mmPhe[cvF$subsets[cvF$which==i],]*probs))
  }
  NNP$CELoss[j]<-mean(CELoss)
}
NNP$nodes<-NNP$nodes*NNP$layers
#plot(NNP$nodes[NNP$layers==1],NNP$CELoss[NNP$layers==1],type="p",ylim=c(0,1.6))
#points(NNP$nodes[NNP$layers==2],NNP$CELoss[NNP$layers==2],type="p",col="blue")
lSmoothDf<-c()
for(i in 1:2)
{
  lSmooth<-ksmooth(x=NNP$nodes[NNP$layers==i],y=NNP$CELoss[NNP$layers==i],bandwidth=20)
  #points(lSmooth$x,lSmooth$y,type="l")
  lSmoothDf<-rbind(lSmoothDf,data.frame(layers=i,
                                        nodes=round(lSmooth$x[which.min(lSmooth$y)]),CELoss=min(lSmooth$y)))
}
NNPBest<-lSmoothDf[which.min(lSmoothDf$CELoss),]
nnetSig<-neuralnet(tempForm,data=DfTrainSig,act.fct="logistic",err.fct="ce",
                   linear.output=FALSE,hidden=rep(round(NNPBest$nodes[1]/NNPBest$layers[1]),times=NNPBest$layers[1]),
                   rep=1,threshold=.00001)

############ Full Comparison ############
# kNN:
knnProbs<-attr(KNN,"prob")
knnProbs<-data.frame(method="kNN",knnProbs*model.matrix(~KNN-1,data.frame(phe=KNN)))
names(knnProbs)<-gsub("KNN","",names(knnProbs))

# kNN Sig:
knnSigProbs<-attr(KNNSig,"prob")
knnSigProbs<-data.frame(method="kNNSig",knnSigProbs*model.matrix(~KNNSig-1,data.frame(phe=KNNSig)))
names(knnSigProbs)<-gsub("KNNSig","",names(knnProbs))

knnRes<-rbind(knnProbs,knnSigProbs)

res<-rbind(knnRes,
           data.frame(method="NB",predict(NB,test,type="raw")),
           data.frame(method="NBSig",predict(NBSig,test,type="raw")),
           data.frame(method="plsda",predict(Plsda,test,type="prob")[,,1]),
           data.frame(method="plsdaSig",predict(PlsdaSig,testSig,type="prob")[,,1]),
           data.frame(method="splsda",predict(Splsda,test,type="prob")),
           data.frame(method="splsdaSig",predict(SplsdaSig,testSig,type="prob")),
           data.frame(method="rF",predict(rF,test,type="prob")),
           data.frame(method="rFSig",predict(rFSig,testSig,type="prob")),
           data.frame(method="SVM",attr(predict(SVM,test,probability=TRUE),"probabilities")),
           data.frame(method="SVMSig",attr(predict(SVMSig,testSig,probability=TRUE),"probabilities")))
res2<-rbind(
  data.frame(method="nnet",
             t(apply(neuralnet::compute(nnet,
                                        covariate=DfTest[,4:ncol(DfTest)])$net.result,1,probNorm))),
  data.frame(method="nnetSig",
             t(apply(neuralnet::compute(nnetSig,
                                        covariate=DfTestSig[,4:ncol(DfTestSig)])$net.result,1,probNorm)))
)
names(res2)<-names(res)
res<-rbind(res,res2)

minProb<-function(x)
{
  minx<-min(x[x>0])
  x[x<=0]<-minx
  return(x)
}
res[,names(res)!="method"]<-apply(res[,names(res)!="method"],2,minProb)

# CE-Loss:
ceLoss<-c()
for(method in unique(res$method))
{
  x<-res[res$method==method,names(res)!="method"]
  ceLoss<-rbind(ceLoss,data.frame(method=method,ceLoss=1/60*sum(-mmPhe*log2(x))))
}

# Misclassification:
res$pred<-levels(pheTrain)[apply(res %>% select(-method),1,which.max)]
res$act<-rep(pheTest,times=14)
res$mis<-as.numeric(res$pred!=res$act)
misRes<-res %>% group_by(method) %>% summarize(mis=mean(mis))

res<-ceLoss %>% left_join(misRes,by="method")
res$selected<-nrow(gr)
res$phe1N<-pheDistribution[1]
res$phe2N<-pheDistribution[2]
res$phe3N<-pheDistribution[3]
save(res,file=paste0("NLres",iter,".RData"))
