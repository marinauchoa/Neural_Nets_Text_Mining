# Introduction to Artificial Neural Networks
# Final Project
# Authors: Amir Golkhari Baghini, Felipe Dutra Calainho and Marina Ferreira Uchoa


#Download and run libraries
lib<-c("tm","plyr","class","fifer","stats","bigmemory","bigpca","Matrix","beepr")
#install.packages(lib)
lapply(lib, require, character.only = T)
rm(lib)

#Import dataset
# The data used can be found and downloaded from Kaggle after according to the competition terms at:
# www.kaggle.com/c/spooky-author-identification/data
# We use the training data set
dataset<-read.csv("IANN_Project_Data.csv")

dataset<-cbind(dataset,class=revalue(as.factor(dataset[,3]),c("EAP"="1", "HPL"="2", "MWS"="3")))

#Clean Dataset
dataset$text<-gsub("[^[:alnum:][:space:]$]","",dataset$text)
dataset$text<-tolower(dataset$text)
dataset$text<-removeWords(dataset$text,stopwords("english"))
dataset$text<-stemDocument(dataset$text, language = "english")
dataset$text<-stripWhitespace(dataset$text)
id<-dataset[,1]

#Take subset for experimentation
# subset<-stratified(dataset, ("class"), size = 30)
# idSub<-subset[,1]

#Build TDM
corpusData <- Corpus(VectorSource(dataset$text))
tdm <- TermDocumentMatrix(corpusData)
tdm <- as.matrix(tdm)
colnames(tdm)<-t(id)

#Build Subset TDM
# corpusDataSub <- Corpus(VectorSource(subset$text))
# tdmSub <- TermDocumentMatrix(corpusDataSub)
# tdmSub <- as.matrix(tdmSub)
# colnames(tdmSub)<-t(idSub)

#Create target vector
targets <- as.data.frame(t(as.numeric(as.character(dataset$class))))
colnames(targets)<-t(id)
#Create Subset target vector
# targetsSub <- as.data.frame(t(as.numeric(as.character(subset$class))))
# colnames(targetsSub)<-t(idSub)

#Remove unnecessary elements from environment
rm(corpusData,i)

#Remove documents with less than 10 words
remove<-c()
for(i in 1:ncol(tdm)){
  remove[i]<-ifelse(sum(tdm[,i])<=10,1,0)
}

#Create new TDM with reduced dataset
# it is not the same as:
# tdm<-tdm[,as.logical(!remove), drop=F]
# because the removed documents had words exclusive to them
sub_dataset<-dataset[as.logical(!remove),]
corpusData <- Corpus(VectorSource(sub_dataset$text))
sub_tdm <- TermDocumentMatrix(corpusData)
sub_tdm <- as.matrix(sub_tdm)
colnames(sub_tdm)<-t(sub_dataset$id)
targets<-targets[as.logical(!remove), drop=F]

save.image(file="project_tdm_red_docs_2.RData")
#Uncomment below if you want a beep to sound when R is finished
#beep()

#Save csv
write.csv(t(sub_tdm),"t_tdm.csv")
write.csv(targets,"targets.csv",row.names = F)
#beep()
#write.csv(tdmSub,"tdmSub.csv")
#write.csv(targetsSub,"targetsSub.csv",row.names = F)

#Save TDM CSV in smaller chunks
write.csv(sub_tdm[,1:5000],"tdm1.csv")
write.csv(sub_tdm[,5001:10000],"tdm2.csv")
write.csv(sub_tdm[,10001:ncol(sub_tdm)],"tdm2.csv")


#Transpose matrix for PCA
t.tdm<-t(sub_tdm)


# Clean unnecessary data from data environment
rm(targets, tdm, tdm_remove, remove, GCtorture, id,remove, corpusData,dataset,sub_dataset,sub_tdm)
#Run PCA
system.time(pca <- prcomp(t.tdm))
#beep()
save.image(file="project_PCA_red_tdm.RData")


#### PRINCIPAL COMPONENTS FULL
PCS<-pca$x
#### Cumulative porportion
cuml_prop<-cumsum(pca$sdev^2 / sum(pca$sdev^2))
which(cumsum(pca$sdev^2 / sum(pca$sdev^2))>=0.9)

#### PRINCIPAL COMPONENTS 1 to 7 (cumulative importance 0.04)
PC7<-PCS[,1:7]
write.csv(PC7,"PC7.csv")
#beep()

#### PRINCIPAL COMPONENTS 1 to 1000 (cumulative importance 0.67)
PC1000<-PCS[,1:1000]
write.csv(PC1000,"PC1000.csv")
#beep()

#### PRINCIPAL COMPONENTS 1 to 2850 (cumulative importance 0.90)
PC2850<-PCS[,1:2850]
write.csv(PC2850,"PC2850.csv")

system.time(cov.x <- cov(x))
x.wt<- rep(1/nrow(x), nrow(x))
system.time(cov.x <- cov.wt(x,wt=x.wt))
system.time(pca <- princomp(x, scores = T))
system.time(pca <- prcomp(t.tdm))

#beep()
save.image(file="project_PCA_6th.RData")
##########
vt<-as.matrix(pca$loadings)
vti<-solve(vt)
