# Flight Delay Prediction with Microsoft R Server

In this example, we use historical on-time performance and weather data to predict whether the arrival of a scheduled passenger flight will be delayed by more than 15 minutes.

We approach this problem as a classification problem, predicting two classes -- whether the flight will be delayed, or whether it will be on time. Broadly speaking, in machine learning and statistics, classification is the task of identifying the class or category to which a new observation belongs, on the basis of a training set of data containing observations with known categories. Classification is generally a supervised learning problem. Since this is a binary classification task, there are only  two classes.

To solve this categorization problem, we build an example using `RevoScaleR` library within Microsoft R Server. The `RevoScaleR` library provides extremely fast statistical analysis on terabyte-class data sets without needing specialized hardware. Also, the `RevoScaleR` platform offers efficient, local data storage, which offers numerous benefits when working with extremely large datasets.

In this example, we train a model using a large number of examples from historic flight data, along with an outcome measure that indicates the appropriate category or class for each example. The two classes are labeled 1 if a flight was delayed, and labeled 0 if the flight was on time.

There are five basic steps in building this example using Microsoft R Server (all the R scripts are available in **MRS_Flight_Delay.R**.):

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
- _Carrier_:	Code assigned by IATA and commonly used to identify a carrier.
- _OriginAirportID_: An identification number assigned by US DOT to identify a unique airport (the flight's origin).
- _DestAirportID_:	An identification number assigned by US DOT to identify a unique airport (the flight's destination).
- _CRSDepTime_: The CRS departure time in local time (hhmm)
- _DepDelay_: Difference in minutes between the scheduled and actual departure times. Early departures show negative numbers.
- _DepDel15_: A Boolean value indicating whether the departure was delayed by 15 minutes or more (1=Departure was delayed)
- _CRSArrTime_: CRS arrival time in local time(hhmm)
- _ArrDelay_: Difference in minutes between the scheduled and actual arrival times. Early arrivals show negative numbers.
- _ArrDel15_: A Boolean value indicating whether the arrival was delayed by 15 minutes or more (1=Arrival was delayed)
- _Cancelled_: A Boolean value indicating whether the arrival flight was cancelled  (1=Flight was cancelled)

**Weather Data.csv** includes the following 14 columns: _AirportID_, _Year_, _AdjustedMonth_, _AdjustedDay_, _AdjustedHour_, _TimeZone_, _Visibility_,  _DryBulbFarenheit_, _DryBulbCelsius_, _DewPointFarenheit_, _DewPointCelsius_, _RelativeHumidity_, _WindSpeed_, _Altimeter_


## <a name="anchor-1"></a> Step 1: Import Data

The `RevoScaleR`, comes with Microsoft R Server (MRS), provides tools for scalable data management and analysis. It contains a wide range of `rx` prefixed functions that include functionality for:
* Accessing external data sets (SAS, SPSS, ODBC, Teradata, and delimited and fixed format text) for analysis in R.
* Efficiently storing and retrieving data in a high performance data file.
* Cleaning, exploring, and manipulating data.
* Fast, basic statistical analysis.

`rxImport()` can import a comma-delimited text file to a `.xdf` file. The `.xdf` data file format is designed for fast processing of blocks of data. The class of `flight` object is `RxXdfData`.
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

After importing the data, we want to examine the information that is contained in the source data. MRS gives us the capability to apply the open source R functions, such as `dim()`, `head()`, `ncol()`, `nrow()`, on the `RxXdfData` class objects.
```
dim(flight)  # 2719418 rows * 14 columns.
head(flight)  # Review the first 6 rows of flight data.
nrow(weather)  # 404914 rows in weather data.
ncol(weather)  # 10 columns in weather data.
```

We can also get `.xdf` File Information by using `rxGetInfo()`.
```
rxGetInfo(flight)
```
![][image1]

And get variable information of flight data by using `rxGetVarInfo()`.
```
rxGetVarInfo(flight)
```
![][image2]

Before performing any statistical analysis or predictive modeling, we want to obtain some basic statistics of the data, for example: **Mean**, **Standard Deviation**, **Minimum**, **Maximum**, and etc.  `rxSummary()` can produce univariate summaries of objects in `RevoScaleR`.
```
rxSummary(~., data = flight, blocksPerRead = 2)
```
![][image3]

If you are familiar with open source R function `summary()`, it can also generate a similar data summary.
```
summary(weather)
```
![][image4]


## <a name="anchor-2"></a> Step 2: Pre-process Data

A dataset usually requires some pre-processing before it can be analyzed.

First, we want to remove some columns that are possible target leakers from the flight data. `varsToDrop` is a character vector of variable names to exclude when reading from the input data file.
```
varsToDrop <- c('DepDelay', 'DepDel15', 'ArrDelay', 'Cancelled', 'Year')
```

We also round down scheduled departure time (`CSRDepTime` column in the flight data) to the full hour so that it can be used as a joining key to concatenate with the weather data. `rxDataStep()` is good function to utilize for data manipulation and can transform data from an input data set to an output data set.
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
* `AdjustedMonth` is renamed as `Month`.
* `AdjustedDay` is renamed as `DayofMonth`.
* `AirportID` is renamed as `OriginAirportID`.
* `AdjustedHour` is renamed as `CRSDepTime`.
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

Now, we can join the flight and weather data using their common keys `Month`, `DayofMonth`, `OriginAirportID`, and `CRSDepTime`. `rxMerge()` can merge variables from two sorted `.xdf` files or data frames on one or more match variables.
* Firstly, join flight records and weather data at origin of the flight (_OriginAirportID_).
```
originData <- rxMerge(inData1 = flight, inData2 = weather, outFile = outFileOrigin,
                      type = 'inner', autoSort = TRUE, decreasing = FALSE,
                      matchVars = c('Month', 'DayofMonth', 'OriginAirportID', 'CRSDepTime'),
                      varsToDrop2 = 'DestAirportID',
                      overwrite=TRUE
                      )
```                  

* Secondly, join flight records and weather data using the destination of the flight (_DestAirportID_).
```
destData <- rxMerge(inData1 = originData, inData2 = weather, outFile = outFileDest,
                    type = 'inner', autoSort = TRUE, decreasing = FALSE,
                    matchVars = c('Month', 'DayofMonth', 'DestAirportID', 'CRSDepTime'),
                    varsToDrop2 = c('OriginAirportID'),
                    duplicateVarExt = c("Origin", "Destination"),
                    overwrite=TRUE
                    )
```

Since some numerical features are not standardized between 0 and 1 scale, we use `scale()` to normalize the column of numeric values. Also, `OriginAirportID` and `DestAirportID` need to be treated as categorical features because each numeric value in those two columns represents different airport.
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

Before choosing and applying a learning algorithm to predict whether the flight will be delayed by more than 15 minutes, we randomly select 80% from the joined dataset to create a training set and use the residual 20% as the test set to evaluate the model obtained from the training set.

`rxExec()` allows distributed execution of a function in parallel across nodes (computers) or cores of a _compute context_ such as a cluster.
```
rxExec(rxSplit, inData = finalData,
       outFilesBase="finalData",
       outFileSuffixes=c("Train", "Test"),
       splitByFactor="splitVar",
       overwrite=TRUE,
       transforms=list(splitVar = factor(sample(c("Train", "Test"), size=.rxNumRows, replace=TRUE, prob=c(.80, .20)),
                       levels= c("Train", "Test"))),
       rngSeed=17,
       consoleOutput=TRUE
       )
```


## <a name="anchor-4A"></a> Step 4A: Choose and apply a learning algorithm (Logistic Regression)

In order to experience different Machine Learning models via MRS, we decide to implement two models to solve this binary classification problem and compare their results.

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

Once we learn the algorithm on the training dataset, we can predict the probability of a flight is going to be delayed by 15 minutes on the test dataset. In the `rxPredict()`, we choose `type = 'response'` because the predictions are on the scale of the response variable in the range of (0, 1).
```
predictLogit <- rxPredict(logitModel, data = 'finalData.splitVar.Test.logit.xdf',
                          type = 'response', overwrite = TRUE)
```
Let's take a look of the first 5 rows of the prediction results. The `ArrDel15_Pred` column contains the predictions.
```
rxGetInfo(predictLogit, getVarInfo = TRUE, numRows = 5)
```
![][image5]

By setting 0.5 as the threshold, we can classify all the predictions that are less than 0.5 as 0 (Arrival was not delayed by 15 minutes) and all the predictions that are greater or equal to 0.5 as 1 (Arrival was delayed by 15 minutes).
```
testDF <- rxImport('finalData.splitVar.Test.logit.xdf')
testDF$ArrDel15_Class[which(testDF$ArrDel15_Pred < 0.5)] <- 0
testDF$ArrDel15_Class[which(testDF$ArrDel15_Pred >= 0.5)] <- 1
```

Let's take a look of the ROC Curve of Logistic Regression model.
```
rxRocCurve( "ArrDel15", "ArrDel15_Pred", predictLogit)
```
![][roc1]

In order to evaluate how this model performs, we calculate the `Area Under the Curve (AUC)`. `AUC` is a metric used to judge predictions in binary response (0 vs. 1) problem. As we can see in the result, the Logistic Regression model has a AUC of **0.70**.
```
rxAuc(rxRoc("ArrDel15", "ArrDel15_Pred", predictLogit))
```

We also compute the `Confusion Matrix` to describe the performance of the Logistic Regression model on a set of test data for which the true values are known.
```
xtab <- table(testDF$ArrDel15_Class, testDF$ArrDel15)
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
predictTree <- rxPredict(dTree2, data = 'finalData.splitVar.Test.tree.xdf',
                         overwrite = TRUE)
```

Again, by setting 0.5 as the threshold, we can classify all the predictions that are less than 0.5 as 0 (Arrival was not delayed by 15 minutes) and all the predictions that are greater or equal to 0.5 as 1 (Arrival was delayed by 15 minutes).
```
testDF2 <- rxImport('finalData.splitVar.Test.tree.xdf')
testDF2$ArrDel15_Class[which(testDF2$ArrDel15_Pred < 0.5)] <- 0
testDF2$ArrDel15_Class[which(testDF2$ArrDel15_Pred >= 0.5)] <- 1
```

Let's take a look of the ROC Curve of Logistic Regression model.
```
rxRocCurve( "ArrDel15", "ArrDel15_Pred", predictTree)
```
![][roc2]

The `AUC` of the Decision Tree model is **0.73**, which is higher than the `AUC` of the Logistic Regression model.
```
rxAuc(rxRoc("ArrDel15", "ArrDel15_Pred", predictTree))
```

The `Confusion Matrix` also shows that the Decision Tree model has a better _Accuracy_ and _Balanced Accuracy_ comparing to the Logistic Regression model when predicting whether the arrival of a scheduled passenger flight will be delayed by more than 15 minutes with these datasets.
```
xtab2 <- table(testDF2$ArrDel15_Class, testDF2$ArrDel15)
confusionMatrix(xtab2, positive = '1')
```
![][image7]



<!-- Images -->
[image1]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image1.PNG
[image2]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image2.PNG
[image3]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image3.PNG
[image4]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image4.PNG
[image5]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image5.PNG
[image6]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image6.PNG
[image7]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/image7.PNG
[roc1]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/roc1.PNG
[roc2]:https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Flight_Delay_Prediction/roc2.PNG
