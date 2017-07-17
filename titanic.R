setwd("~/Desktop/kaggle/titanic")
train <- read.csv("~/Desktop/kaggle/titanic/train.csv")
test <- read.csv("~/Desktop/kaggle/titanic/test.csv")

View(train)
str(train)
#creates table
table(train$Survived)
#creates table of proportions 38% survived, 62% died
prop.table(table(train$Survived))

#create column of 'everyone dies in test df bc most died in traindf
test$Survived <- rep(0, 418)

#kaggle submission for all die
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "theyallperish.csv", row.names = FALSE)

#refine it based on survival of males vs females
summary(train$Sex)
#1=rows, 2=columns
prop.table(table(train$Sex, train$Survived), 1)
#everyone dies
test$Survived <- 0
#...except females
test$Survived[test$Sex == 'female'] <- 1

#kaggle submission for all men die all females survive
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "allmalesperish.csv", row.names = FALSE)

#refine it based on age of passengers
#create new column: 0 = adult, 1=child
train$Child <- 0
train$Child[train$Age < 18] <- 1

#does that rly say fun ha
aggregate(Survived ~ Child + Sex, data=train, FUN=sum)
#aggregate for proportion
aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)})
#trend of females survive, males die even applies to children

#look at fares, bin fares by amount
train$Fare2 <- '30+'
train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
train$Fare2[train$Fare < 10] <- '<10'

aggregate(Survived ~ Fare2 + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
#use insight to set females in 3rd class with high ticket price to 0
test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$Fare >= 20] <- 0

#kaggle submission for new insights
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "try3.csv", row.names = FALSE)

#DECISON TREES to automate this#
library(rpart)
#build the model
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data=train,
             method="class")

#install packages to better view the tree
fancyRpartPlot(fit)

#new submission for kaggle based on d tree predictions
Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "myfirstdtree.csv", row.names = FALSE)

#FEATURE ENGINEERING
train$Name[1]

#create column in test set of 'NA's
#combine train and test for feature engineering
test$Survived <- NA
test$Fare2 <- NA
test$Child <- 0
combi <- rbind(train, test)

#names from factors to string
combi$Name <- as.character(combi$Name)
combi$Name[1]
#string split to parse names, create new column
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
#remove space before title
combi$Title <- sub(' ', '', combi$Title)

table(combi$Title)
#combine low occuring titles into larger categories
combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady' 

#change title back to a factor
combi$Title <- factor(combi$Title)

#create family size variable (+1 is for the person named)
combi$FamilySize <- combi$SibSp + combi$Parch + 1

combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
table(combi$FamilyID)
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]

combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)

#break apart train and test set 
train <- combi[1:891,]
test <- combi[892:1309,] 

#fit the model
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
             data=train, 
             method="class") 
fancyRpartPlot(fit) 

#new submission for kaggle with the feature engineering
Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "secondtree.csv", row.names = FALSE)

summary(combi$Age)
#263 values out of 1309 were missing this whole time
#fill in NAs with random forest - insanity!
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combi[!is.na(combi$Age),], 
                method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

#a bit more cleaning
summary(combi$Embarked)
which(combi$Embarked == '')
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)

summary(combi$Fare)
which(is.na(combi$Fare))
combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)

#break apart train and test set 
train <- combi[1:891,]
test <- combi[892:1309,] 

#Random Forest
install.packages('randomForest')
library(randomForest)

#set the seed 
#use the same seed number each time so that the same random numbers are generated inside the Random Forest
set.seed(415)

fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                      Embarked + Title + FamilySize + FamilyID2,
                    data=train, 
                    importance=TRUE, 
                    ntree=2000)

varImpPlot(fit)

#submission for kaggle...didn't score as high as simpler decision tree
Prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE) 

#try conditional decision tree
install.packages('party')
library(party)

#use same seed
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                   Embarked + Title + FamilySize + FamilyID,
                 data = train, 
                 controls=cforest_unbiased(ntree=2000, mtry=3))

Prediction <- predict(fit, test, OOB=TRUE, type = "response")

#final submission for kaggle
Prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "final.csv", row.names = FALSE)
