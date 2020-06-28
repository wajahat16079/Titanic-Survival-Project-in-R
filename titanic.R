rm(list = ls())
library(forecast)
library(neuralnet)
library(e1071)
library(plotly)
library(reshape)
library(GGally)
library(Amelia)
library(caret)
library(ggplot2)
setwd("C:/Users/student/Desktop/R/titanic")

train = read.csv("train.csv", stringsAsFactors = FALSE)
test = read.csv("test.csv", stringsAsFactors = FALSE)

  
# matching columns number on both sets of data
test$Survived = NA
## creating a new dataset 'full'
full = rbind(test, train)
summary(full)
names(full)
str(full)

# looking at possibe features which can be converted to factors
apply(full,2, function(x) length(unique(x)))

# Converting the features Survived, Pclass, Sex and Embarked to factors
cols<-c("Survived","Pclass","Sex","Embarked")



for (i in cols){
  full[,i] <- as.factor(full[,i])
}


# looking for any Missing values
missmap(full)


colSums(is.na(full))
# Age Fare have NAs

colSums(full=="")
# cabin and Embarked have empty strings


summary(full$Ticket)
head(full$Ticket)
tail(full$Ticket)
# Ticket seems to have random aplpha numeric code so it will be removed 
# cabin has a lot of missing values so we will remove it from our data sets
# name and passengerid will also be removed 
full = subset( full, select = -c(Cabin,Ticket,Name, PassengerId))

str(full)

# 

##Cleaning Data

## filling out NAs and other missing values
## Ticket col has some aplphanumeric code but doesnt seems to have 
#any specific meaning attached to it  
## Cabin col has alot of "" so we will be ignoring it.
## summary shows Age and fare have some NAs
## where embarked has two empty strings

## assigning the mode of emabarked to missing embarked
table(full$Embarked == '')
full[full$Embarked == '',"Embarked"] = "S" 
table(full$Embarked == '')




# assigning mean of fare to the missing values
summary(full$Fare)
full[is.na(full$Fare),"Fare"] = mean(full$Fare, na.rm = TRUE) 
summary(full$Fare)

# finding out missing age through SVM
names(titanic)
# age has nothing to do with passenger id and name,where survived is still unknown
# splitting the data into two data sets  

have_age = subset(full,is.na(Age) == FALSE)
predict_age = subset(full, is.na(Age) == TRUE )

smp_size <- floor(0.80 * nrow(have_age))
train_ind <- sample(seq_len(nrow(have_age)), size = smp_size)
train_age <- have_age[train_ind, ]
test_age <- have_age[-train_ind,]
names(train_age)

# since Age has NAs we will not pass it in our train data set 
svm_model_age = svm(Age~Pclass+Sex+SibSp+Parch+Fare+Embarked, data = subset(train_age, select = -Survived ),
                             type = "eps-regression", kernel = "radial")


test_age$age_predicted = predict(svm_model_age, subset(test_age, select = -Survived ))
accuracy(test_age$Age, test_age$age_predicted)

predict_age$Age = predict(svm_model_age, subset(predict_age, select = -c(Age,Survived) ))

#combining the two data, full1 doesnt have any missing value.   
full1 = rbind(have_age, predict_age)



# looking for any Missing values
missmap(full1)
colSums(is.na(full1))
# only Age has NAs

colSums(full1=="")
# no empty strings found

## we have clean data set now

## dividing the data into two sets
have_survived = subset(full1,is.na(Survived) == FALSE)
predict_survived = subset(full1, is.na(Survived) == TRUE )

## Visual Analysis


# Analyzing the role of gender in Survival
ggplot(have_survived,aes(x=Sex,fill=Survived))+geom_bar()
# a female has more chances of survival compare to a male

# Analyzing the role of Pclass in Survival
ggplot(have_survived,aes(x=Pclass,fill=Survived))+
  geom_bar(position = "fill")+
  ylab("Frequency")
# chances of survival are higher in class 1 and least in class 3

# looking at gender classwise
ggplot(data = have_survived,aes(x=Pclass,fill=Survived))+
  geom_bar(position="fill")+
  facet_wrap(~Sex)
# a female has higher chances of survival compared to a man regardless of class 


# Analyzing the role of Sibsp in Survival
ggplot(have_survived,aes(x=SibSp,fill=Survived))+geom_bar()

# Analyzing the role of Parch in Survival
ggplot(have_survived,aes(x=Parch,fill=Survived))+geom_bar()


# parch and SibSp seems to have similar impact on survivor
# but we are not sure if SibSp 0 corresponds to same passenger in Parch 0


# Analyzing the role of Embarked in Survival
ggplot(have_survived,aes(x=Embarked,fill=Survived))+
  geom_bar(position = "fill")+
  ylab("Frequency")
# S and Q have little below 50% survived
# C has a little above 50% survived

# Analyzing the role of Age in Survival
ggplot(data = have_survived,aes(x=Age,fill=Survived))+
  geom_histogram(binwidth = 3,position="fill")+
  ylab("Frequency")
# Children aged below 15 and old people aged above 80 have more chances of survival

# Analyzing the role of Fare in Survival
ggplot(data = have_survived,aes(x=Fare,fill=Survived))+
  geom_histogram(binwidth =20, position="fill")

# chances of survival increase with increasing fare
ggplot(data = have_survived,aes(x=Fare,fill=Pclass))+
  geom_histogram(binwidth =20, position="fill")
# class one has the highest fare and, class 3 has least

names(have_survived)
## two datsets have_survived, predict_survived


smp_size <- floor(0.80 * nrow(have_survived))
train_ind <- sample(seq_len(nrow(have_survived)), size = smp_size)
train_survived <- have_survived[train_ind, ]
test_survived <- have_survived[-train_ind,]
names(train_survived)

# since Age has NAs we will not pass it in our train data set 

## predicting with glm

glm_model_survived = glm(Survived~.,family = "binomial",
                         data = train_survived)
test_survived$predicted_survived = predict(glm_model_survived,test_survived)
test_survived$predicted_survived = ifelse(test_survived$predicted_survived > 0.5,1,0)
test_survived$predicted_survived = as.factor(test_survived$predicted_survived)
confusionMatrix(test_survived$predicted_survived,test_survived$Survived)

## predictiing with SVM
svm_model_survived = svm(Survived~., data = train_survived,
                    type = "C-classification", kernel = "radial")

test_survived$predicted_survived = predict(svm_model_survived,test_survived)
confusionMatrix(test_survived$predicted_survived,test_survived$Survived)


## using most accurate of the models above to predict 
predict_survived$Survived = predict(svm_model_survived,subset(predict_survived, select = -Survived ))

## Visual Analysis of Predicted Data


# Analyzing the role of gender wrt predicted Survived
ggplot(predict_survived,aes(x=Sex,fill=Survived))+geom_bar()
# as expected a female has more chances of survival compare to a male

# Analyzing the role of Pclass wrt predicted Survived
ggplot(predict_survived,aes(x=Pclass,fill=Survived))+
  geom_bar(position = "fill")+
  ylab("Frequency")
# as expected chances of survival are higher in class 1 and least in class 3

# looking at gender classwise
ggplot(data = predict_survived,aes(x=Pclass,fill=Survived))+
  geom_bar(position="fill")+
  facet_wrap(~Sex)
# as expected a female has higher chances of survival compared to a man regardless of class 



# Analyzing the role of Age wrt predicted Survived
ggplot(data = predict_survived,aes(x=Age,fill=Survived))+
  geom_histogram(binwidth = 3,position="fill")+
  ylab("Frequency")
# As expected Children and old people have higher chances of survival

# Analyzing the role of Fare wrt predicted Survived
ggplot(data = predict_survived,aes(x=Fare,fill=Survived))+
  geom_histogram(binwidth =20, position="fill")
# as expected chances of survival increase with increasing fare
