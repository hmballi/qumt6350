#Chapter 14 - HW 5
#Group 4: Hilary Balli, Marco Duran Perez, George Garcia

library(caret)
library(rpart)
library(C50)
library(AppliedPredictiveModeling)

data(churn)

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
ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = defaultSummary,
                     sampling = "smote", 
                     classProbs = TRUE,
                     savePredictions = TRUE)

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
cartTune #Accuracy = .84, Kappa = .48


### Plot the tuning results
plot(cartTune, scales = list(x = list(log = 10)))  

library(partykit)
 
cartTree <- as.party(cartTune$finalModel)
plot(cartTree)


cartImp <- varImp(cartTune, scale = FALSE, competes = FALSE)
cartImp

### Save the test set results in a data frame                 
testResults <- data.frame(obs = yTest,
                          CART = predict(cartTune, xTest))
testResults

### Tune the conditional inference tree

cGrid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 2))))

class(xTrain[,-20])
class(yTrain)

set.seed(seed)
ctreeTune <- train(x = xTrain[,-20], y = yTrain,
                   method = "ctree",
                   tuneGrid = cGrid,
                   trControl = ctrl)
ctreeTune
plot(ctreeTune) #errors in class between old data and new data?

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

ctreebag #accuracy = .85, kappa = .50

#random forest
set.seed(seed)
crandomforest <- caret::train(x = xTrain[, -20], y = yTrain, 
                       method = "rf", 
                       preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                       trControl = ctrl)
crandomforest #accuracy = .89, kappa = .53

#boosted trees
set.seed(seed)
cgbm <- caret::train(x = xTrain[, -20], y = yTrain, 
              method = "gbm", 
              preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
              trControl = ctrl, verbose = FALSE)
cgbm #accuracy = .84, kappa = .44

#c5.0
str(x)
??churn
?c5.0
?C5.0.formula
set.seed(seed)
c50Grid <- expand.grid(trials = c(1:5), 
                       model = c("tree", "rules"),
                       winnow = c(TRUE, FALSE))
set.seed(seed)
c50Fit <- train(x = xTrain[, -20],
                y = yTrain,
                method = "C5.0",
                preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                tuneGrid = c50Grid,
                verbose = FALSE,
                metric = "Accuracy",
                trControl = ctrl)
c50Fit

c50Fit$pred <- merge(c50Fit$pred,  c50Fit$bestTune)
c50Fit$pred
c50CM <- confusionMatrix(c50Fit, norm = "none")
c50CM

#OR

oneTree <- C5.0(x = xTrain[,-20], y = yTrain,
              method = "c5.0",
              preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
              trControl = ctrl)
oneTree
#issues here, incorrect # of dimensions                
oneTreePred <- predict(oneTree, y)
oneTreeProbs <- predict(oneTree, y, type = "prob")
postResample(oneTreePred, y$churn)
  
#boosted c5.0
set.seed(seed)
C5Boost <- C5.0(x = xTrain[, -20],
                 y = yTrain,
                 method = "C5.0",
                 trials = 50, 
                 preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                 tuneGrid = c50Grid,
                 verbose = FALSE,
                 metric = "Accuracy",
                 trControl = ctrl)
C5Boost

summary(C5Boost)
plot(C5Boost)


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
# Finalize Model
library(Cubist)
# prepare the data transform using training data
set.seed(seed)
x <- xTrain[,-20]
y <- yTrain
preprocessParams <- preProcess(x, method = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"))
trans_x <- predict(preprocessParams, x)


# train the final model
finalModel <- cubist(x = trans_x, y = y, committees = 18)
summary(finalModel)

# transform the test dataset
set.seed(seed)
val_x <- xTest[,-20]
trans_val_x <- predict(preprocessParams, val_x)
val_y <- yTest

# use final model to make predictions on the test dataset
predictions <- predict(finalModel, newdata=trans_val_x, neighbors=3)

# calculate RMSE
rmse <- RMSE(predictions, val_y)
print(rmse)

#caret ensemble
# define training control
control <- trainControl(method="repeatedcv", number=10, repeats=3, 
                        index=createResample(xTrain, 10),
                        savePredictions="final")


# train a list of models
library(caretEnsemble)

algorithmList <- c('lm', 'glm', 'svmRadial','rpart', 'knn', 'ridge','rf','gbm','cubist' )


models <- caretList(x = xTrain[, -20], y = yTrain, 
                    trControl=control,
                    preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                    methodList=algorithmList)

results<-resamples(models)

summary(results)
dotplot(results)
modelCor(results)

 
model_list= c( 'glm', 'rpart', 'knn', 'rf','gbm')

models <- caretList(x = xTrain[, -20], y = yTrain, 
                    trControl=control,
                    preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                    methodList=model_list)


##############caretStack #######################

# stack using glm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3,
                             savePredictions=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="glm",  
                        preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                        trControl=stackControl)
print(stack.glm)
summary(stack.glm)

# stack using random forest
set.seed(seed)
stack.rf <- caretStack(models, method="rf",
                       preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                       trControl=stackControl)
print(stack.rf)
summary(stack.rf)

# stack using rpart
set.seed(seed)
stack.rpart <- caretStack(models, method="rpart",
                       preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                       trControl=stackControl)
print(stack.rpart)
summary(stack.rpart)

# stack using gbm #For combining, using ensemble methods is not recommended becasue of overfitting.
set.seed(seed)
stack.gbm <- caretStack(models, method="gbm",
                           preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                           trControl=stackControl)
print(stack.gbm)
summary(stack.gbm)

# stack using cubist # For combining, using ensemble methods is not recommended becasue of overfitting.
set.seed(seed)
stack.cubist <- caretStack(models, method="cubist",
                          preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                          trControl=stackControl)
print(stack.cubist)
summary(stack.cubist)

# use final model to make predictions on the test dataset
greedy_ensemble <- caretEnsemble(models, metric="RMSE",
                                 trControl=trainControl(number=6))

summary(greedy_ensemble)


# (a)
predictions <- predict(greedy_ensemble, newdata=trans_val_x)

# calculate RMSE
rmse <- RMSE(predictions, val_y)
print(rmse)

#(b)
predictions <- predict(stack.glm, newdata=trans_val_x)

# calculate RMSE
rmse <- RMSE(predictions, val_y)
print(rmse)

#(c)
predictions <- predict(stack.rf , newdata=trans_val_x)

# calculate RMSE
rmse <- RMSE(predictions, val_y)
print(rmse)

#(d)
predictions <- predict(stack.rpart , newdata=trans_val_x)

# calculate RMSE
rmse <- RMSE(predictions, val_y)
print(rmse)

#(e)
predictions <- predict(stack.cubist , newdata=trans_val_x)

# calculate RMSE
rmse <- RMSE(predictions, val_y)
print(rmse)

# (f)
predictions <- predict(stack.gbm , newdata=trans_val_x)

# calculate RMSE
rmse <- RMSE(predictions, val_y)
print(rmse)
