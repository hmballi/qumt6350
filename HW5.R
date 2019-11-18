#Chapter 14 - HW 5
#Group 4: Hilary Balli, Marco Duran Perez, George Garcia

library(caret)
library(rpart)
library(C50)
library(AppliedPredictiveModeling)

data(churn)
??churn

# Assign variables and combine
xTrain <- churnTrain[, -20]
yTrain <- churnTrain[, 20]
xTest <- churnTest[, -20]
yTest <- churnTest[, 20]
x <- rbind(xTrain, xTest)
y <- c(yTrain,yTest)

# Set seed number
seed <- 127

# Model Training
# Controlled resampling and subsampling
set.seed(seed)
indx <- createFolds(yTrain, returnTrain = TRUE) ##returnTrain ; When true, the values returned are the sample positions corresponding to the data used during training.
ctrl <- trainControl(method = "cv", index = indx)

trainData <- xTrain
trainData$y <- yTrain

rpStump <- rpart(y ~ ., data = trainData, 
                 control = rpart.control(maxdepth = 1))
rpSmall <- rpart(y ~ ., data = trainData, 
                 control = rpart.control(maxdepth = 2))

### Tune the model

set.seed(seed)
cartTune <- train(x = xTrain[,-20], y = yTrain,
                  method = "rpart",
                  tuneLength = 25,
                  metric = "Accuracy",
                  trControl = ctrl)
cartTune #Accuracy = .93, Kappa = .68
## cartTune$finalModel


### Plot the tuning results
plot(cartTune, scales = list(x = list(log = 10)))  

### Use the partykit package to make some nice plots. First, convert
### the rpart objects to party objects.

library(partykit)
# 
cartTree <- as.party(cartTune$finalModel)
plot(cartTree)

### Get the variable importance. 'competes' is an argument that
### controls whether splits not used in the tree should be included
### in the importance calculations.

cartImp <- varImp(cartTune, scale = FALSE, competes = FALSE)
cartImp

### Save the test set results in a data frame                 
testResults <- data.frame(obs = yTest,
                          CART = predict(cartTune, xTest))
testResults

### Tune the conditional inference tree

cGrid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 2))))

set.seed(seed)
ctreeTune <- train(x = xTrain, y = yTrain,
                   method = "ctree",
                   tuneGrid = cGrid,
                   trControl = ctrl)
ctreeTune
plot(ctreeTune)

##ctreeTune$finalModel               
plot(ctreeTune$finalModel)

testResults$cTree <- predict(ctreeTune, xTest)

#bagged trees
set.seed(seed)
ctreebag <- caret::train(x = xTrain[, -20], y = yTrain,
                  method = "treebag",
                  preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                  nbagg = 50,  
                  trControl = ctrl)

ctreebag

#random forest
set.seed(seed)
crandomforest <- caret::train(x = xTrain[, -20], y = yTrain, 
                       method = "rf", 
                       preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                       trControl = ctrl)
crandomforest
#boosted trees
set.seed(seed)
cgbm <- caret::train(x = xTrain[, -20], y = yTrain, 
              method = "gbm", 
              preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
              trControl = ctrl, verbose = FALSE)
cgbm

#c5.0
oneTree <- C5.0(
#boosted c5.0

# eXtreme Gradient Boosting Tree 
#control function

ctrl2 <- trainControl(method = "LGOCV", 
                     summaryFunction = defaultSummary, 
                     sampling = "smote", 
                     classProbs = TRUE,
                     savePredictions = TRUE)

set.seed(seed)
modelXGBT <- train(x = xTrain[, -20], y = yTrain, 
                   method = "xgbTree",
                   preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                   trControl = ctrl2)
modelXGBT #RMSE = 


 # eXtreme Gradient Boosting
set.seed(seed)
modelXGBD <- train(x = xTrain[,-20], y = yTrain, 
                   method = "xgbDART", 
                   preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                   trControl = ctrl2)
modelXGBD #RMSE = 


set.seed(seed)
modelXGBL <- train(x = xTrain[,-20], y = yTrain, 
                   method = "xgbLinear",
                   preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                   trControl = ctrl2)
modelXGBL #RMSE = , R^2 = , MAE = 

#catboost
library(catboost)
set.seed(seed)
churncatboost <- caret::train(x = xTrain, y = yTrain,
                              method = "catboost.caret",
                              trcontrol = ctrl2,
                              tunegrid = param,
                              metric = "ROC") 

#caret ensemble
