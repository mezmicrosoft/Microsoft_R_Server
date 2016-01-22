# Flight Delay Prediction using Microsoft R Server (a.k.a. Revolution R Enterprise)

In this example, we use historical on-time performance and weather data to predict whether the arrival of a scheduled passenger flight will be delayed by more than 15 minutes.

We approach this problem as a classification problem, predicting two classes -- whether the flight will be delayed, or whether it will be on time. Broadly speaking, in machine learning and statistics, classification is the task of identifying the class or category to which a new observation belongs, on the basis of a training set of data containing observations with known categories. Classification is generally a supervised learning problem. Since this is a binary classification task, there are only  two classes.

To solve this categorization problem, we will build an example using Microsoft R Server. In this example, we train a model using a large number of examples from historic flight data, along with an outcome measure that indicates the appropriate category or class for each example. The two classes are labeled 1 if a flight was delayed, and labeled 0 if the flight was on time.

There are five basic steps in building this example using Microsoft R Server (all the R scripts are available in **RRE_Flight_Delay.R**.):

Prepare the Data

- [Step 1: Import Data](#anchor-1)
- [Step 2: Pre-process Data](#anchor-2)
- [Step 3: Prepare Training and Test Datasets](#anchor-3)

Train the Model

- [Step 4A: Choose and apply a learning algorithm (Logistic Regression)](#anchor-4A)
- [Step 4B: Choose and apply a learning algorithm (Decision Tree)](#anchor-4B)

Score and Test the Model

- [Step 5A: Predict over new data (Logistic Regression)](#anchor-5A)
- [Step 5B: Predict over new data (Decision Tree)](#anchor-5B)

------------------------------------------

## Data

**Flight Delays Data.csv** (unzip the _Flight Delays Data.zip_ file) includes the following 14 columns: _Year_, _Month_, _DayofMonth_, _DayOfWeek_, _Carrier_, _OriginAirportID_, _DestAirportID_, _CRSDepTime_, _DepDelay_, _DepDel15_, _CRSArrTime_, _ArrDelay_, _ArrDel15_, and _Cancelled_.

These columns contain the following information:
- _Carrier_ -	Code assigned by IATA and commonly used to identify a carrier.
- _OriginAirportID_ - An identification number assigned by US DOT to identify a unique airport (the flight's origin).
- _DestAirportID_ -	An identification number assigned by US DOT to identify a unique airport (the flight's destination).
- _CRSDepTime_ - The CRS departure time in local time (hhmm)
- _DepDelay_ - Difference in minutes between the scheduled and actual departure times. Early departures show negative numbers.
- _DepDel15_ -	 A Boolean value indicating whether the departure was delayed by 15 minutes or more (1=Departure was delayed)
- _CRSArrTime_ - CRS arrival time in local time(hhmm)
- _ArrDelay_ - Difference in minutes between the scheduled and actual arrival times. Early arrivals show negative numbers.
- _ArrDel15_ - A Boolean value indicating whether the arrival was delayed by 15 minutes or more (1=Arrival was delayed)
- _Cancelled_ - A Boolean value indicating whether the arrival flight was cancelled  (1=Flight was cancelled)

**Weather Data.csv** includes the following 14 columns: _AirportID_, _Year_, _AdjustedMonth_, _AdjustedDay_, _AdjustedHour_, _TimeZone_, _Visibility_,  _DryBulbFarenheit_, _DryBulbCelsius_, _DewPointFarenheit_, _DewPointCelsius_, _RelativeHumidity_, _WindSpeed_, _Altimeter_


## <a name="anchor-1"></a> Step 1: Import Data

The `RevoScaleR`, comes with Microsoft R Server, provides tools for scalable data management and analysis. It contains a wide range of `rx` prefixed functions that include functionality for:
1. Accessing external data sets (SAS, SPSS, ODBC, Teradata, and delimited and fixed format text) for analysis in R.
2. Efficiently storing and retrieving data in a high performance data file.
3. Cleaning, exploring, and manipulating data.
4. Fast, basic statistical analyses.

`rxImport()` can import a comma-delimited text file to a `.xdf` file. The `.xdf` data file format, designed for fast processing of blocks of data. The class of `flight` object is `RxXdfData`.
```
# Import the flight data.
flight <- rxImport(inData = inputFileFlight, outFile = outFileFlight,
                  missingValueString = "M", stringsAsFactors = TRUE)

# Import the weather dataset and eliminate some features due to redundance.
weather <- rxImport(inData = inputFileWeather, outFile = outFileWeather,
                    missingValueString = "M", stringsAsFactors = TRUE,
                    varsToDrop = c('Year', 'Timezone', 'DryBulbFarenheit', 'DewPointFarenheit'),
                    overwrite=TRUE)
```
Open source R functions, such as `dim()`, `head()`, `ncol()`, `nrow()`, can also be applied to `RxXdfData` class objects.

Now, let's examine the imported datasets.
```
dim(flight)  # 2719418 rows * 14 columns.
head(flight)  # Review the first 6 rows of flight data.
nrow(weather)  # 404914 rows in weather data.
ncol(weather)  # 10 columns in weather data.
```

We can also get .xdf File Information by using `rxGetInfo()`.
```
rxGetInfo(flight)
```
![][image1]

And get variable information of flight data by using `rxGetVarInfo`.
```
rxGetVarInfo(flight)
```
![][image2]

Summary the flight data is easy by using `rxSummary()`.
```
rxSummary(~., data = flight, blocksPerRead = 2)
```
![][image3]

The Open Source R `summary()` can also generate a similar data summary.
```
summary(weather)
```
![][image4]


## <a name="anchor-2"></a> Step 2: Pre-process Data

A dataset usually requires some pre-processing before it can be analyzed.

First, we remove columns that are possible target leakers from the flight data. `varsToDrop` is a character vector of variable names to exclude when reading from the input data file.
```
varsToDrop <- c('DepDelay', 'DepDel15', 'ArrDelay', 'Cancelled', 'Year')
```

We also round down scheduled departure time (`CSRDepTime` column in the flight data) to the full hour so that it can be used as a joining key to concatenate with the weather data. `rxDataStep()` can transform data from an input data set to an output data set.
```
xform <- function(dataList) {
  # Create a new continuous variable from an existing continuous variables:
  # round down CSRDepTime column to the nearest hour.
  dataList$CRSDepTime <- sapply(dataList$CRSDepTime,
                                FUN = function(x) {floor(x/100)})

  # Return the adapted variable list.
  return(dataList)
}
flight <- rxDataStep(inData = flight,
                           outFile = outFileFlightS2,
                           varsToDrop = varsToDrop,
                           transformFunc = xform,
                           transformVars = 'CRSDepTime',
                           overwrite=TRUE
                           )
```

To prepare the data for the merging later, we rename some column names in the weather data.
```
xform2 <- function(dataList) {
  # Create a new column 'DestAirportID' in weather data.
  dataList$DestAirportID <- dataList$AirportID
  # Rename 'AdjustedMonth', 'AdjustedDay', 'AirportID', 'AdjustedHour'.
  names(dataList)[match(c('AdjustedMonth', 'AdjustedDay', 'AirportID', 'AdjustedHour'),
                 names(dataList))] <- c('Month', 'DayofMonth', 'OriginAirportID', 'CRSDepTime')

  # Return the adapted variable list.
  return(dataList)
}
weather <- rxDataStep(inData = weather,
                      outFile = outFileWeather2,
                      transformFunc = xform2,
                      transformVars = c('AdjustedMonth', 'AdjustedDay', 'AirportID', 'AdjustedHour'),
                      overwrite=TRUE
                      )
```            
Then we can join flight and weather data using keys `Month`, `DayofMonth`, `OriginAirportID`, and `CRSDepTime`. `rxMerge()` can merge variables from two sorted `.xdf` files or data frames on one or more match variables.
1. Join flight records and weather data at origin of the flight (OriginAirportID).
```
originData <- rxMerge(inData1 = flight, inData2 = weather, outFile = outFileOrigin,
                      type = 'inner', autoSort = TRUE, decreasing = FALSE,
                      matchVars = c('Month', 'DayofMonth', 'OriginAirportID', 'CRSDepTime'),
                      varsToDrop2 = 'DestAirportID',
                      overwrite=TRUE
                      )
```                  

2. Join flight records and weather data using the destination of the flight (DestAirportID).
```
destData <- rxMerge(inData1 = originData, inData2 = weather, outFile = outFileDest,
                    type = 'inner', autoSort = TRUE, decreasing = FALSE,
                    matchVars = c('Month', 'DayofMonth', 'DestAirportID', 'CRSDepTime'),
                    varsToDrop2 = c('OriginAirportID'),
                    duplicateVarExt = c("Origin", "Destination"),
                    overwrite=TRUE
                    )
```

Since some numerical features are not standerlized between 0 and 1 scale, we use `scale()` to normalize the column of numeric values. Also, `OriginAirportID` and `DestAirportID` need to be treated as categorical features because each numeric value in those two columns represents different airport.
```
finalData <- rxDataStep(inData = destData, outFile = outFileFinal,
                      transforms = list(
                                        # Normalize some numerical features
                                        Visibility.Origin = scale(Visibility.Origin),
                                        DryBulbCelsius.Origin = scale(DryBulbCelsius.Origin),
                                        DewPointCelsius.Origin = scale(DewPointCelsius.Origin),
                                        RelativeHumidity.Origin = scale(RelativeHumidity.Origin),
                                        WindSpeed.Origin = scale(WindSpeed.Origin),
                                        Altimeter.Origin = scale(Altimeter.Origin),
                                        Visibility.Destination = scale(Visibility.Destination),
                                        DryBulbCelsius.Destination = scale(DryBulbCelsius.Destination),
                                        DewPointCelsius.Destination = scale(DewPointCelsius.Destination),
                                        RelativeHumidity.Destination = scale(RelativeHumidity.Destination),
                                        WindSpeed.Destination = scale(WindSpeed.Destination),
                                        Altimeter.Destination = scale(Altimeter.Destination),

                                        # Convert 'OriginAirportID', 'DestAirportID' to categorical features
                                        OriginAirportID = factor(OriginAirportID),
                                        DestAirportID = factor(DestAirportID)
                                        ),
                      overwrite=TRUE
                      )
```


## <a name="anchor-3"></a> Step 3: Prepare Training and Test Datasets

Before choosing and applying a learning algorithm to predict whether the flight will be delayed by more than 15 minutes, we randomly split 80% data as training set and the remaining 20% as test set. `rxExec()` allows distributed execution of a function in parallel across nodes (computers) or cores of a _compute context_ such as a cluster.
```
rxExec(rxSplit, inData = finalData,
       outFilesBase="finalData",
       outFileSuffixes=c("Train", "Test"),
       splitByFactor="splitVar",
       overwrite=TRUE,
       transforms=list(splitVar = factor(sample(c("Train", "Test"), size=.rxNumRows, replace=TRUE, prob=c(.80, .10)),
                       levels= c("Train", "Test"))),
       rngSeed=17,
       consoleOutput=TRUE
       )
```


## <a name="anchor-4A"></a> Step 4A: Choose and apply a learning algorithm (Logistic Regression)

Since this example is a binary classification problem, we decide to use two different classification models to solve this problem and compare their results.

The first model we build is a Logistic Regression model using the `rxLogit()` function. Since the formula `dot(.)` expansion is currently not supported, we need to build the formula between `ArrDel15` and other independent variables as below.
```
# Build the formula.
allvars <- names(finalData)
xvars <- allvars[allvars !='ArrDel15']
form <- as.formula(paste("ArrDel15", "~", paste(xvars, collapse = "+")))  

# Build a Logistic Regression model.
logitModel <- rxLogit(form, data = 'finalData.splitVar.Train.xdf')
summary(logitModel)
```


## <a name="anchor-5A"></a> Step 5A: Predict over new data (Logistic Regression)

Once we learn the algorithm on the training dataset, we can predict the probability of a flight is going to delay on the test dataset. In the `rxPredict`, we choose `type = 'response'` because the predictions are on the scale of the response variable in the range of (0, 1).
```
predictLogit <- rxPredict(logitModel, data = 'finalData.splitVar.Test.xdf',
                          outData = 'logitTest.xdf',
                          type = 'response', overwrite = TRUE)
```
Let's take a look of the first 5 rows of the prediction results. The `ArrDel15_Pred` column contains the predictions.
```
rxGetInfo(predictLogit, getVarInfo = TRUE, numRows = 5)
```
![][image5]

By setting 0.5 as the threshold, we can classify all the predictions that are less than 0.5 as 0 (Arrival was not delayed) and all the predictions that are greater or equal to 0.5 as 1 (Arrival was delayed).
```
testDF <- rxImport('finalData.splitVar.Test.xdf')
predictDF <- rxImport('logitTest.xdf')
predictDF$ArrDel15_Class[which(predictDF$ArrDel15_Pred < 0.5)] <- 0
predictDF$ArrDel15_Class[which(predictDF$ArrDel15_Pred >= 0.5)] <- 1
```

In order to evaluate how the model performs, we calculate the `Area Under the Curve (AUC)`. `AUC` is a metric used to judge predictions in binary response (0/1) problem. As we can see in the result, the Logistic Regression model has a AUC of 0.6998.
```
auc <- function(outcome, prob){
  N <- length(prob)
  N_pos <- sum(outcome)
  df <- data.frame(out = outcome, prob = prob)
  df <- df[order(-df$prob),]
  df$above <- (1:N) - cumsum(df$out)
  return( 1- sum( df$above * df$out ) / (N_pos * (N-N_pos) ) )
}
auc(testDF$ArrDel15, predictDF$ArrDel15_Pred)
```

We also compute the Confusion Matrix to describe the performance of the Logistic Regression model on a set of test data for which the true values are known.
```
xtab <- table(predictDF$ArrDel15_Class, testDF$ArrDel15)
(if(!require("e1071")) install.packages("e1071"))
(if(!require("caret")) install.packages("caret"))
library(e1071)
library(caret)
confusionMatrix(xtab, positive = '1')
```
![][image6]


## <a name="anchor-4B"></a> Step 4B: Choose and apply a learning algorithm (Decision Tree)

After building the Logistic Regression model, we are also interested in seeing how a Decision Tree model would perform on this dataset.

First, we build a basic Decision Tree model with all default parameters.
```
dTree1 <- rxDTree(form, data = 'finalData.splitVar.Train.xdf')
```

Then, we want to find the best value of `cp` for pruning a `rxDTree` object. `cp` is a numeric scalar that specifies the complexity parameter. Any split that does not decrease overall lack-of-fit by at least `cp` is not attempted.
```
treeCp <- rxDTreeBestCp(dTree1)
```

Once we have the best value of `cp` that is determined by `rxDTreeBestCp()`, we prune a decision tree created by `rxDTree()` and return the smaller tree.
```
dTree2 <- prune.rxDTree(dTree1, cp = treeCp)
```


## <a name="anchor-5B"></a> Step 5B: Predict over new data (Decision Tree)

We predict the probability of flight delay on the test dataset using the trained Decision Tree model.
```
predictTree <- rxPredict(dTree2, data = 'finalData.splitVar.Test.xdf',
                         outData = 'dTreeTest.xdf',
                         overwrite = TRUE)
```

Again, by setting 0.5 as the threshold, we can classify all the predictions that are less than 0.5 as 0 (Arrival was not delayed) and all the predictions that are greater or equal to 0.5 as 1 (Arrival was delayed).
```
predictDF2 <- rxImport('dTreeTest.xdf')
predictDF2$ArrDel15_Class[which(predictDF2$ArrDel15_Pred < 0.5)] <- 0
predictDF2$ArrDel15_Class[which(predictDF2$ArrDel15_Pred >= 0.5)] <- 1
```

The `AUC` of the Decision Tree model is 0.7284, which is higher than the `AUC` of the Logistic Regression model.
As we can see in the result, the Logistic Regression model has a AUC of 0.6998.
```
auc(testDF$ArrDel15, predictDF2$ArrDel15_Pred)
```

The Confusion Matrix also shows that the Decision Tree model has a better _Accuracy_ and _Balanced Accuracy_ comparing to the Logistic Regression model when predicting whether the arrival of a scheduled passenger flight will be delayed by more than 15 minutes with these datasets.
```
xtab2 <- table(predictDF2$ArrDel15_Class, testDF$ArrDel15)
confusionMatrix(xtab2, positive = '1')
```
![][image7]


**Microsoft R Server** is fun to play with and works extrmely well with large-scale datasets. When you're looking for a solution to deal with over million of records, you can definitely give a try on Microsoft R Server.


<!-- Images -->
[image1]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image1.PNG
[image2]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image2.PNG
[image3]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image3.PNG
[image4]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image4.PNG
[image5]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image5.PNG
[image6]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image6.PNG
[image7]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image7.PNG
