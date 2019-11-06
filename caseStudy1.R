library(AppliedPredictiveModeling)
library(stats)
library(caret)
library(corrplot)
library(e1071)
library(mlbench)

data("abalone")
?abalone
head(abalone)
summary(abalone)
str(abalone)
sapply(abalone, attributes)
sapply(abalone, class)

#set outcome variable
predictors <- subset(abalone, select = -8)
rings <- subset(abalone, select = "Rings")

#skewness
skew <- lapply(predictors, skewness)
head(skew)

#set training and testing data
set.seed(127)
atrainrows <- createDataPartition(rings$Rings,
                                  p = .7,
                                  list = FALSE)
atestpredictors <- predictors[-atrainrows,]
testRings <- rings[-atrainrows,]

atrainpredictors <- predictors[atrainrows,]
trainRings <- rings[atrainrows,]

library(skimr)
skimmed1 <- skim_to_wide(atrainpredictors)
skimmed2 <- skim_to_wide(trainRings)

#histogram
hist(rings$Rings, breaks = 100, col = "blue")

#density plot
plot(density(rings$Rings))

#boxplot
boxplot(rings$Rings, data = rings, main = "Rings")

#correlation plot
correlations <- cor(atrainpredictors[sapply(atrainpredictors, function(x) !is.factor(x))])
corrplot(correlations, method = "number")

highcorrelation <- findCorrelation(correlations, cutoff = .8)
length(highcorrelation)
head(highcorrelation)
atrainfiltered <- atrainpredictors[, -highcorrelation]
atestfiltered <- atestpredictors[, -highcorrelation]
atrainfiltered[1:20,]
atestfiltered[1:20,]

#create control for 10 fold cross validation
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Note: Data cleaning operations are applied in this order: zero-variance filter, near-zero variance filter, 
# correlation filter, Box-Cox/Yeo-Johnson/exponential transformation, centering, scaling, range, imputation,
# PCA, ICA then spatial sign. (R Documentation, Pre-Processing of Predictors, Package caret v. 6.0-84)

#linear regression
set.seed(127)
lmtune <- train(x = atrainpredictors, y = trainRings,
                method = "lm",
                preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                trControl = control)
lmtune
testresults <- data.frame(obs = testRings, 
                          linear_regression = predict(lmtune, atestpredictors))

#partial least squares
set.seed(127)
plstune <- train(x = atrainpredictors, y = trainRings,
                 method = "pls",
                 tuneGrid = expand.grid(ncomp = 1:8),
                 trControl = control)

plstune

testresultspls <- predict(plstune, atestpredictors)

set.seed(127)
pcrtune <- train(x = atrainpredictors, y = trainRings,
                 method = "pcr",
                 tuneGrid = expand.grid(ncomp = 1:8),
                 trControl = control)
pcrtune

plsresamples <- plstune$results
plsresamples$model <- "pls"
pcrresamples <- pcrtune$results
pcrresamples$model <- "pcr"
plsPlotData <- rbind(plsresamples, pcrresamples)

xyplot(RMSE ~ncomp,
       data = plsPlotData,
       #aspect = 1,
       xlab = "Number of Componenets",
       ylab = "RMSE",
       auto.key = list(columns = 2),
       type = c("o", "g"))
plsimp <- varImp(plstune, scale = FALSE)
plot(plsimp, top = 25, scale = list(y = list(cex = .95)))

# Oscorepls fit
set.seed(127)
apls <- train(x = atrainpredictors, y = trainRings,
              method = "pls",
              tuneLength = 100,
              trControl = control)
plstunepcrtuneapls
summary(apls) # RMSE = 

# Simpls fit
set.seed(127)
apls2 <- train(x = atrainpredictors, y = trainRings,
               method = "simpls",
               tuneLength = 100,
               trControl = control)
apls2
summary(apls2) # RMSE = 0.0000058

# Widekernelpls fit
set.seed(127)
apls3 <- train(x = data.matrix(atrainpredictors, rownames.force = NA), y = trainRings,
               method = "widekernelpls",
               tuneLength = 100,
               trControl = control)
apls3
summary(apls3) # RMSE = 


#ridge regression
ridgegrid <- expand.grid(lambda = seq(0, 0.1, length = 15))
set.seed(127)
ridgetune <- train(x = atrainpredictors[, -1], y = trainRings,
                          method = "ridge",
                          tuneGrid = ridgegrid,
                          trControl = control)
ridgetune
print(update(plot(ridgetune)))
summary(ridgetune)

#lasso regression
set.seed(127)
lassotune <- train(x = atrainpredictors[, -1], y = trainRings,
                   method = "lasso",
                   trControl = control,
                   preProc = c("center", "scale", "YeoJohnson","corr", "spatialSign", "knnImpute", "zv", "nzv"))
lassotune
plot(lassotune)
summary(lassotune)
#elastic regression
enetgrid <- expand.grid(lambda = c(0, 0.01, .1),
                        fraction = seq(.05, 1, length = 20))
set.seed(127)
enettune <- train(x = atrainpredictors[, -1], y = trainRings,
                  method = "enet",
                  tuneGrid = enetgrid,
                  trControl = control,
                  preProc = c("center", "scale", "YeoJohnson","corr", "spatialSign", "knnImpute", "zv", "nzv"))
enettune
plot(enettune)
summary(enettune)

#MARS
set.seed(127)
amars <- train(x = atrainpredictors, y = trainRings,
               method = "earth",
               tuneGrid = expand.grid(degree = 1:2, nprune = 2:38),
               trControl = control)
amars
plot(amars)
summary(amars)

#support vector machine
set.seed(127)
asvmr <- train(x = atrainpredictors[, -1], y = trainRings,
               method = "svmRadial",
               tuneLength = 100,
               trControl = control,
               preProc = c("center", "scale", "YeoJohnson"))
asvmr
summary(asvmr)

set.seed(127)
asvmp <- train(x = atrainpredictors[, -1], y = trainRings,
               method = "svmPoly",
               tuneLength = 100,
               trControl = control,
               preProc = c("center", "scale", "YeoJohnson"))
asvmp
summary(asvmp)

#k-nearest neighbors
set.seed(127)
aknn <- train(x = atrainpredictors[, -1], y = trainRings,
              method = "knn",
              preProc = c("center", "scale", "YeoJohnson"),
              trControl = control)
aknn
summary(aknn)

#CART
set.seed(127)
grid <- expand.grid(.cp = c(0, 0.05, 0.1))
acart <- train(x = atrainpredictors, y = trainRings, 
               method = "rpart",  
               tuneGrid = grid, 
               preProc = c("center", "scale", "YeoJohnson","corr", "spatialSign","knnImpute", "zv", "nzv"  ), 
               trControl = control)
acart
summary(acart)

library(party)
library(partykit)
library(ipred)
library(randomForest)
library(Cubist)
library(gbm)
library(doMC)

#bagged tree
set.seed(127)
atreebag <- train(x = atrainpredictors, y = trainRings,
                  method = "treebag",
                  nbagg = 50,  
                  trControl = control)

atreebag


#random forests
set.seed(127)
arandomforest <- train(x = atrainpredictors, y = trainRings, 
                       method = "rf", 
                       preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                       trControl = control)
arandomforest

#boosted trees
set.seed(127)
agbm <- train(x = atrainpredictors, y = trainRings, 
              method = "gbm", 
              preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
              trControl = control, verbose = FALSE)
agbm

#gradient boosting machine

#cubist
set.seed(127)
acubist <- train(x = atrainpredictors, y = trainRings, 
                 method = "cubist", 
                 preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                 trControl = control)
acubist



# Compare algorithms using resamples()
transform_results <- resamples(list(LM=lmtune, PLS=plstune, PCR=pcrtune, APLS=apls))

summary(transform_results)
dotplot(transform_results)
