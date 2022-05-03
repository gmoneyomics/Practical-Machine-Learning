Machine Learning Assignment
================
Ariel Gershman
5/3/2022

## Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

Data The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>.

First we will load in the data and using the carat package split the data into the training set and test set (70% for training, 30% for test).

``` r
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
setwd("~/CourseHera/Machine_Learning/")
trainURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainURL))
testing <- read.csv(url(testURL))

label <- createDataPartition(training$classe, p = 0.7, list = FALSE)
train <- training[label, ]
test <- training[-label, ]
```

The dataset has 160 variables, but the ones with little variance or that contain NA are not useful to us, so we remove these. We removed values close to zero and variables where variance is less than .90. This reduces the number of variables to 54.

``` r
nonZero <- nearZeroVar(train)
train <- train[ ,-nonZero]
test <- test[ ,-nonZero]

lab <- apply(train, 2, function(x) mean(is.na(x))) > 0.90
train <- train[, -which(lab, label == FALSE)]
test <- test[, -which(lab, label == FALSE)]

train <- train[ , -(1:5)]
test <- test[ , -(1:5)]
```

## Data Exploration

Now that the dataset is cleaned, we look at how all these variables correlate with one another through a correlation plot.

    ## corrplot 0.92 loaded

![](Practical-Machine-Learning-assigment1_files/figure-markdown_github/corplot-1.png) Figure 1. A correlation plot between all the variables. The more blue the more highly correlated variables are.

## Prediction Model Selection

We will test 3 different models (Decision Tree, Random Forest and Generalized Bootested Model). By comparing these models we will determine which is the most accruate and use a confusion matrix to visualize how well the model classifies.

# 1. Random Forest

``` r
# model fit
set.seed(12312)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=train, method="rf",
                          trControl=controlRF)
modFitRandForest$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = min(param$mtry, ncol(x))) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.23%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3904    1    0    0    1 0.0005120328
    ## B    6 2648    3    1    0 0.0037622272
    ## C    0    4 2391    1    0 0.0020868114
    ## D    0    0   11 2241    0 0.0048845471
    ## E    0    0    0    4 2521 0.0015841584

``` r
# prediction on Test dataset
predictRandForest <- predict(modFitRandForest,test)
confMatRandForest <- confusionMatrix(predictRandForest, as.factor(test$classe))
confMatRandForest
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    4    0    0    0
    ##          B    0 1135    3    0    0
    ##          C    0    0 1023    2    0
    ##          D    0    0    0  962    3
    ##          E    0    0    0    0 1079
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.998           
    ##                  95% CI : (0.9964, 0.9989)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9974          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9965   0.9971   0.9979   0.9972
    ## Specificity            0.9991   0.9994   0.9996   0.9994   1.0000
    ## Pos Pred Value         0.9976   0.9974   0.9980   0.9969   1.0000
    ## Neg Pred Value         1.0000   0.9992   0.9994   0.9996   0.9994
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1929   0.1738   0.1635   0.1833
    ## Detection Prevalence   0.2851   0.1934   0.1742   0.1640   0.1833
    ## Balanced Accuracy      0.9995   0.9979   0.9983   0.9987   0.9986

# 2. Decision Tree

``` r
library(rpart)
library(rpart.plot)
library(rattle)
```

    ## Loading required package: tibble

    ## Loading required package: bitops

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.5.1 Copyright (c) 2006-2021 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

``` r
set.seed(342142)
DTmod <- rpart(classe ~ ., data = train, method = "class")
fancyRpartPlot(DTmod)
```

![](Practical-Machine-Learning-assigment1_files/figure-markdown_github/DescTree-1.png)

``` r
DTpred <- predict(DTmod, test, type = "class")
DTconfmat <- confusionMatrix(DTpred, as.factor(test$classe))
DTconfmat
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1419   87    4   12    4
    ##          B   98  818   47   55   85
    ##          C   39  101  817   47   24
    ##          D  115  117  134  791  164
    ##          E    3   16   24   59  805
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7901          
    ##                  95% CI : (0.7795, 0.8005)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.7358          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8477   0.7182   0.7963   0.8205   0.7440
    ## Specificity            0.9746   0.9399   0.9566   0.8923   0.9788
    ## Pos Pred Value         0.9299   0.7416   0.7947   0.5988   0.8875
    ## Neg Pred Value         0.9415   0.9329   0.9570   0.9621   0.9444
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2411   0.1390   0.1388   0.1344   0.1368
    ## Detection Prevalence   0.2593   0.1874   0.1747   0.2245   0.1541
    ## Balanced Accuracy      0.9111   0.8291   0.8764   0.8564   0.8614

# 3. Generatlized Boosted Model

``` r
set.seed(51354)
cont <- trainControl(method = "repeatedcv", number = 5, repeats = 1, verboseIter = FALSE)
GBM <- train(classe ~ ., data = train, trControl = cont, method = "gbm", verbose = FALSE)
GBM$finalModel
```

    ## A gradient boosted model with multinomial loss function.
    ## 150 iterations were performed.
    ## There were 53 predictors of which 52 had non-zero influence.

The Random Forest offers the highest accuracy (however, takes the longest time to compute). Therefore, we will use the Random Forest.

## Prediction

``` r
RF <- predict(modFitRandForest, testing)
RF
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
