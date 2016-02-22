# Practical Machine Learning Course Project
Joe Guarascio  
February 22, 2016  

##Introduction
This project was done for the Practical Machine Learning course from Coursera.  The goal of the project is to create a model to predict the manner in which various exercises were done based on accelerometer data.

The data come from 6 participants who wore accelerometers on the belt, forearm, arm, and dumbell and were asked to perform barbell lifts correctly and incorrectly in 5 different ways:  

* exactly according to the specification (Class A)
* throwing the elbows to the front (Class B)
* lifting the dumbbell only halfway (Class C)
* lowering the dumbbell only halfway (Class D)
* throwing the hips to the front (Class E)

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

* Training Data:  https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
* Test Data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

To begin, load the caret library and set the random number seed in order to make the outcomes exactly reproducible.


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(12345)
```

##Data
Load the training data and split it into a training set (75%) and validation set (25%).


```r
training <- read.csv("pml-training.csv")
inTrain <- createDataPartition(y=training$classe, p=0.75, list=FALSE)
training.t <- training[inTrain, ]
training.v <- training[-inTrain, ]
```

##Features
Not all features of the data can be used for prediction.  Features that have near zero variance or that contain a lot of NAs cannot be used.  In addition the first 5 columns of the data are used to identify the subject and recording times, and cannot be used as predictors.  This step will remove these features from the dataset. Features identified from the training set will also be removed from the validation set for consistency.


```r
# remove features with nearly zero variance
nzv <- nearZeroVar(training.t)
training.t <- training.t[, -nzv]
training.v <- training.v[, -nzv]

# remove any features that are NA 95% of the time
mostlyNA <- sapply(training.t, function(x) mean(is.na(x))) > 0.95
training.t <- training.t[, mostlyNA==F]
training.v <- training.v[, mostlyNA==F]

# remove the first 5 features dealing with the user and recording time
training.t <- training.t[, -(1:5)]
training.v <- training.v[, -(1:5)]
```

##Algorithm
Try 3 different models using 3-fold cross validation.


```r
tc <- trainControl(method="cv", number=3, verboseIter=FALSE)

#Basic Decision Tree
rpart<-train(classe~.,data=training.t,method="rpart",trControl=tc) 

#Linear Discriminate Analysis
lda<-train(classe~.,data=training.t,method="lda",trControl=tc) 

#Random Forests
rf<-train(classe~.,data=training.t,method="rf",trControl=tc) 
```


##Evaluation
Use each model to predict on the validation set and evaluate the accuracy of each.


```r
predrpart <- predict(rpart, newdata=training.v)
predlda <- predict(lda, newdata=training.v)
predrf <- predict(rf, newdata=training.v)


confusionMatrix(training.v$classe, predrpart)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1302   20   68    0    5
##          B  253  335  361    0    0
##          C  157   24  674    0    0
##          D  145  131  498    0   30
##          E   35   82  246    0  538
## 
## Overall Statistics
##                                          
##                Accuracy : 0.581          
##                  95% CI : (0.567, 0.5948)
##     No Information Rate : 0.3858         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.4626         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6882  0.56588   0.3649       NA   0.9389
## Specificity            0.9691  0.85761   0.9408   0.8361   0.9162
## Pos Pred Value         0.9333  0.35300   0.7883       NA   0.5971
## Neg Pred Value         0.8319  0.93502   0.7103       NA   0.9913
## Prevalence             0.3858  0.12072   0.3766   0.0000   0.1168
## Detection Rate         0.2655  0.06831   0.1374   0.0000   0.1097
## Detection Prevalence   0.2845  0.19352   0.1743   0.1639   0.1837
## Balanced Accuracy      0.8286  0.71174   0.6529       NA   0.9276
```

```r
confusionMatrix(training.v$classe, predlda)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1161   32  106   88    8
##          B  141  611  123   42   32
##          C   88   76  580   88   23
##          D   32   52  108  579   33
##          E   48  131   79   90  553
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7104          
##                  95% CI : (0.6975, 0.7231)
##     No Information Rate : 0.2998          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6334          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7898   0.6774   0.5823   0.6528   0.8521
## Specificity            0.9319   0.9155   0.9296   0.9440   0.9182
## Pos Pred Value         0.8323   0.6438   0.6784   0.7201   0.6138
## Neg Pred Value         0.9119   0.9264   0.8973   0.9249   0.9760
## Prevalence             0.2998   0.1839   0.2031   0.1809   0.1323
## Detection Rate         0.2367   0.1246   0.1183   0.1181   0.1128
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.8608   0.7965   0.7560   0.7984   0.8851
```

```r
confusionMatrix(training.v$classe, predrf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    3  941    5    0    0
##          C    0    2  852    1    0
##          D    0    0    3  801    0
##          E    0    0    0    4  897
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9963          
##                  95% CI : (0.9942, 0.9978)
##     No Information Rate : 0.2851          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9954          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9979   0.9979   0.9907   0.9938   1.0000
## Specificity            1.0000   0.9980   0.9993   0.9993   0.9990
## Pos Pred Value         1.0000   0.9916   0.9965   0.9963   0.9956
## Neg Pred Value         0.9991   0.9995   0.9980   0.9988   1.0000
## Prevalence             0.2851   0.1923   0.1754   0.1644   0.1829
## Detection Rate         0.2845   0.1919   0.1737   0.1633   0.1829
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9989   0.9979   0.9950   0.9965   0.9995
```

Three models were evaluated:  decision trees, linear discriminate analysis, and random forests.  The model with the highest accuracy is random forests.  The accuracy is very close to 100% (.9963) meaning that the out of sample error is 0. 


## Predicting on the Test Set
First, preprocess and re-fit the random forests model using the full training set to improve accuracy even further.


```r
# remove features with nearly zero variance
nzv <- nearZeroVar(training)
training <- training[, -nzv]

# remove any features that are NA 95% of the time
mostlyNA <- sapply(training, function(x) mean(is.na(x))) > 0.95
training <- training[, mostlyNA==F]

# remove the first 5 features dealing with the user and recording time
training <- training[, -(1:5)]

# re-fit model using full training set
tc <- trainControl(method="cv", number=3, verboseIter=FALSE)
fit <- train(classe~., data=training, method="rf", trControl=tc)
```


Finally, load the testing data and use the model to predict the outcomes:

```r
# Load the testing data
testing <- read.csv("pml-testing.csv")

# Output the final predictions
predict(fit, newdata=testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
