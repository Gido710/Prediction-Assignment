ML Activity Measure CEProject
================

## Load data

First we load the libraries that are needed to perform the analysis and
read the data as the data frame object. Next we investigate the names of
the data set.

``` r
library(caret)
```

    ## Ładowanie wymaganego pakietu: ggplot2

    ## Ładowanie wymaganego pakietu: lattice

``` r
library(gbm)
```

    ## Loaded gbm 2.1.9

    ## This version of gbm is no longer under development. Consider transitioning to gbm3, https://github.com/gbm-developers/gbm3

``` r
test_file <- "pml-testing.csv"
training_file <- "pml-training.csv"

data_test <- read.csv(test_file)
data_train <- read.csv(training_file)

tail(names(data_train),20)
```

    ##  [1] "var_accel_forearm"    "avg_roll_forearm"     "stddev_roll_forearm" 
    ##  [4] "var_roll_forearm"     "avg_pitch_forearm"    "stddev_pitch_forearm"
    ##  [7] "var_pitch_forearm"    "avg_yaw_forearm"      "stddev_yaw_forearm"  
    ## [10] "var_yaw_forearm"      "gyros_forearm_x"      "gyros_forearm_y"     
    ## [13] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
    ## [16] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
    ## [19] "magnet_forearm_z"     "classe"

## Preprocessing

As we see there are plenty of the columns that might not be useful in
our prediction. There is 160 feathers. So we will cutoff the columns
only for these that are related with a specific activity and person.
There was a problem later in the pipeline that ML functions gave the
error, so we also try to omit all the data that are *NA* - it solved the
problem. Lets keep in mind that that all this will reduce the number of
feathers.

``` r
data_train$classe <- factor(data_train$classe)
selector1 <- colnames(data_test)[(sapply(colnames(data_test),
                                   function(x){
                                     (sum(is.na(data_test[,x]))==0)
                                   }))][1:59]
selector1 <- c(selector1,'classe')
data_train <- data_train[ , selector1 ]
selector2 <- colnames(data_train)[grep('_belt|_arm|_dumb|_forearm|classe',colnames(data_train))]
data_train <- data_train[ , selector2 ]
length(selector2)
```

    ## [1] 53

This will give us 53 feathers. Next we divide the data set into training
and testing.

``` r
selectors <- createDataPartition( y=data_train$classe, p = 0.6 , list = FALSE )
train_set <- data_train[ selectors, ]
test_set <- data_train[ -selectors, ]
dim(train_set)
```

    ## [1] 11776    53

We choose to have 11776 ( 60% ) of the data into training and 40% into
testing.

## Machine Learning

In this section we will train 3 models:

- RF - Random Forest
- GBM - Generalized Boosted Regression Models
- LDA - Linear Discriminant Analysis

``` r
model1 <- train(classe~., data = train_set, method = 'rf')
model2 <- train(classe~., data = train_set, method = 'gbm',verbose=0)
model3 <- train(classe~., data = train_set, method = 'lda',verbose=0)
```

## Predictions

The we do the predictions on our test set make the confusion matrix to
investigate which one is the best.

``` r
preds1 <- predict( model1, data = test_set )
preds2 <- predict( model2, data = test_set )
preds3 <- predict( model3, data = test_set )

conf1 <- confusionMatrix( preds1, train_set$classe )
conf2 <- confusionMatrix( preds2, train_set$classe )
conf3 <- confusionMatrix( preds3, train_set$classe )
```

After performing prediction we check the confusion matrix for:

- conf1 - Random Forest
- conf2 - Generalized Boosted Regression Models
- conf3 - Linear Discriminant Analysis

``` r
conf1
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3348    0    0    0    0
    ##          B    0 2279    0    0    0
    ##          C    0    0 2054    0    0
    ##          D    0    0    0 1930    0
    ##          E    0    0    0    0 2165
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9997, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##                                      
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

``` r
conf2
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3315   52    0    1    3
    ##          B   27 2194   36    6    4
    ##          C    4   31 1993   49   13
    ##          D    2    1   24 1869   18
    ##          E    0    1    1    5 2127
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9764          
    ##                  95% CI : (0.9735, 0.9791)
    ##     No Information Rate : 0.2843          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9701          
    ##                                           
    ##  Mcnemar's Test P-Value : 8.74e-07        
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9901   0.9627   0.9703   0.9684   0.9824
    ## Specificity            0.9934   0.9923   0.9900   0.9954   0.9993
    ## Pos Pred Value         0.9834   0.9678   0.9536   0.9765   0.9967
    ## Neg Pred Value         0.9961   0.9911   0.9937   0.9938   0.9961
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2815   0.1863   0.1692   0.1587   0.1806
    ## Detection Prevalence   0.2863   0.1925   0.1775   0.1625   0.1812
    ## Balanced Accuracy      0.9917   0.9775   0.9802   0.9819   0.9909

``` r
conf3
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2749  341  193  116   75
    ##          B   79 1462  201   74  365
    ##          C  268  284 1363  221  200
    ##          D  240   89  249 1446  207
    ##          E   12  103   48   73 1318
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7081          
    ##                  95% CI : (0.6997, 0.7163)
    ##     No Information Rate : 0.2843          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6306          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8211   0.6415   0.6636   0.7492   0.6088
    ## Specificity            0.9140   0.9243   0.8999   0.9203   0.9754
    ## Pos Pred Value         0.7913   0.6703   0.5835   0.6481   0.8481
    ## Neg Pred Value         0.9278   0.9149   0.9268   0.9493   0.9171
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2334   0.1242   0.1157   0.1228   0.1119
    ## Detection Prevalence   0.2950   0.1852   0.1984   0.1895   0.1320
    ## Balanced Accuracy      0.8675   0.7829   0.7818   0.8347   0.7921

## Conclusion

We that the best accuracy has the **model1** - random forest.
