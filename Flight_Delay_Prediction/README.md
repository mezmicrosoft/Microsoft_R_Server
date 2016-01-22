# Flight Delay Prediction using Microsoft R Server (a.k.a. Revolution R Enterprise)

In this example, we use historical on-time performance and weather data to predict whether the arrival of a scheduled passenger flight will be delayed by more than 15 minutes.

We approach this problem as a classification problem, predicting two classes -- whether the flight will be delayed, or whether it will be on time. Broadly speaking, in machine learning and statistics, classification is the task of identifying the class or category to which a new observation belongs, on the basis of a training set of data containing observations with known categories. Classification is generally a supervised learning problem. Since this is a binary classification task, there are only  two classes.

To solve this categorization problem, we will build an example using Microsoft R Server. In this example, we train a model using a large number of examples from historic flight data, along with an outcome measure that indicates the appropriate category or class for each example. The two classes are labeled 1 if a flight was delayed, and labeled 0 if the flight was on time.

There are five basic steps in building an experiment in Azure ML Studio:

Prepare the Data

- [Step 1: Import Data](#anchor-1)
- [Step 2: Pre-process Data](#anchor-2)
- [Step 3: Prepare Training and Test Datasets](#anchor-3)

Train the Model

- [Step 4: Choose and apply a learning algorithm](#anchor-4)

Score and Test the Model

- [Step 5: Predict over new data](#anchor-5)

------------------------------------------

## Data

**Flight Delays Data.csv** includes the following 14 columns: _Year_, _Month_, _DayofMonth_, _DayOfWeek_, _Carrier_, _OriginAirportID_, _DestAirportID_, _CRSDepTime_, _DepDelay_, _DepDel15_, _CRSArrTime_, _ArrDelay_, _ArrDel15_, and _Cancelled_.

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
- _Cancelled_ - A Boolean value indicating whether the arrivalflight was cancelled  (1=Flight was cancelled)

We also used a set of weather observations: **[Hourly land-based weather observations from NOAA](http://cdo.ncdc.noaa.gov/qclcd_ascii/).**

**Weather Data.csv** includes the following 14 columns: _AirportID_, _Year_, _AdjustedMonth_, _AdjustedDay_, _AdjustedHour_, _TimeZone_, _Visibility_,  _DryBulbFarenheit_, _DryBulbCelsius_, _DewPointFarenheit_, _DewPointCelsius_, _RelativeHumidity_, _WindSpeed_, _Altimeter_


## <a name="anchor-1"></a> Step 1: Import Data

The `RevoScaleR`, comes with Microsoft R Server, provides tools for scalable data management and analysis. It contains a wide range of `rx` functions that include functionality for:
1. Accessing external data sets (SAS, SPSS, ODBC, Teradata, and delimited and fixed format text) for analysis in R.
2. Efficiently storing and retrieving data in a high performance data file.
3. Cleaning, exploring, and manipulating data.
4. Fast, basic statistical analyses.

`rxImport` can a import comma-delimited text file to a `.xdf` file. The `.xdf` data file format, designed for fast processing of blocks of data. The class of `flight` object is `RxXdfData`.
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
Open source R functions, such as `dim()`, `head()`, `ncol()`, `nrow()`, `summary()`, can also be applied to `RxXdfData` class objects.

Examine the imported datasets.
```
dim(flight)  # 2719418 rows * 14 columns.
head(flight)  # Review the first 6 rows of flight data.
nrow(weather)  # 404914 rows in weather data.
ncol(weather)  # 10 columns in weather data.
```
Get .xdf File Information.
```
rxGetInfo(flight)
```
![][image1]
Get variable information of flight data.
```
rxGetVarInfo(flight)
```
![][image2]
Summary the flight data.
```
rxSummary(~., data = flight, blocksPerRead = 2)
```
![][image3]
Summary the weather data.
```
summary(weather)
```
![][image4]


## <a name="anchor-2"></a> Step 2: Pre-process Data

Remove columns that are possible target leakers from the flight data. `varsToDrop` character vector of variable names to exclude when reading from the input data file.
```
varsToDrop <- c('DepDelay', 'DepDel15', 'ArrDelay', 'Cancelled', 'Year')
```

Round down scheduled departure time (`CSRDepTime` column in the flight data) to the full hour so that it can be used as a joining key to concatenate with the weather data. `rxDataStep` can transform data from an input data set to an output data set.
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




## <a name="anchor-3"></a> Step 3: Prepare Training and Test Datasets

## <a name="anchor-4"></a> Step 4: Choose and apply a learning algorithm

## <a name="anchor-5"></a> Step 5: Predict over new data

A dataset usually requires some pre-processing before it can be analyzed.

![][image8]


**Flight Data Preprocessing**

First, we used the [**Project Columns**](https://msdn.microsoft.com/library/azure/1ec722fa-b623-4e26-a44e-a50c6d726223) module to exclude from the dataset columns that are possible target leakers: _DepDelay_, _DepDel15_, _ArrDelay_, _Cancelled_, _Year_.
![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight1.PNG)

The columns _Carrier_, _OriginAirportID_, and _DestAirportID_ represent categorical attributes. However, because they are integers, they are initially parsed as continuous numbers; therefore, we used the [**Metadata Editor**](https://msdn.microsoft.com/library/azure/370b6676-c11c-486f-bf73-35349f842a66) module to convert them to categorical.


![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight2.PNG)


We need to join the flight records with the hourly weather records, using the scheduled departure time as one of the join keys. To do this, the _CSRDepTime_ column must be rounded down to the nearest hour using two successive instances of the [**Apply Math Operation**](https://msdn.microsoft.com/library/azure/6bd12c13-d9c3-4522-94d3-4aa44513af57) module.
![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight6.PNG)

**Weather Data Preprocessing**

Columns that have a large proportion of missing values are excluded using the [**Project Columns**](https://msdn.microsoft.com/library/azure/1ec722fa-b623-4e26-a44e-a50c6d726223) module. These include all string-valued columns: _ValueForWindCharacter_, _WetBulbFarenheit_, _WetBulbCelsius_, _PressureTendency_, _PressureChange_, _SeaLevelPressure_, and _StationPressure_.

![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight7.PNG) The [**Clean Missing Data**](https://msdn.microsoft.com/library/azure/d2c5ca2f-7323-41a3-9b7e-da917c99f0c4) module is then applied to the remaining columns to remove rows with missing data.

The time of the weather observation is rounded up to the nearest full hour, so that the column can be equi-joined with the scheduled flight departure time. Note that the scheduled flight time and the weather observation times are rounded in opposite directions. This is done to ensure that the model uses only weather observations that happened in the past, relative to flight time. Also note that the weather data is reported in local time, but the origin and destination may be in different time zones. Therefore, an adjustment to time zone difference must be made by subtracting the time zone columns from the scheduled departure time (_CRSDepTime_) and weather observation time (_Time_). These operations are done using the [**Execute R Script**](https://msdn.microsoft.com/en-us/library/azure/dn905952.aspx) module.


<!--
![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight9.PNG)
![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight10.PNG)
-->

The resulting columns are _Year_, _AdjustedMonth_, _AdjustedDay_, _AirportID_, _AdjustedHour_, _Timezone_, _Visibility_, _DryBulbFarenheit_, _DryBulbCelsius_, _DewPointFarenheit_, _DewPointCelsius_, _RelativeHumidity_, _WindSpeed_, _Altimeter_.

**Joining Datasets**

Flight records are joined with weather data at origin of the flight (_OriginAirportID_) by using the [**Join**](https://msdn.microsoft.com/library/azure/124865f7-e901-4656-adac-f4cb08248099) module.

![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v2/4/flight11.PNG)

<!--
![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight12.PNG) ![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight13.PNG)
-->

Flight records are joined with weather data using the destination of the flight (_DestAirportID_).

![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v2/4/flight14.PNG)

**Preparing Training and Validation Samples**

The training and validation samples are created by using the [**Split**](https://msdn.microsoft.com/library/azure/70530644-c97a-4ab6-85f7-88bf30a8be5f) module to divide the data into April-September records for training, and October records for validation.

![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight15.PNG)

Year and month columns are  removed from the training dataset using the [**Project Columns**](https://msdn.microsoft.com/library/azure/1ec722fa-b623-4e26-a44e-a50c6d726223) module.
The training data is then separated into equal-height bins using the [**Quantize Data**](https://msdn.microsoft.com/library/azure/61dd433a-ee80-4ac3-87f0-b54708644d93) module, and the same binning method was applied to the validation data.
![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight16.PNG)

The training data is split once more, into a training dataset and an optional validation dataset.

![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight17.PNG)

## <a name="anchor-3"></a> Define Features
In machine learning, *features* are individual measurable properties of something you’re interested in. Finding a good set of features for creating a predictive model requires experimentation and knowledge about the problem at hand. Some features are better for predicting the target than others. Also, some features have a strong correlation with other features, so they will not add much new information to the model and can be removed. In order to build a model, we can use all the features available, or we can select a subset of the features in the dataset. Typically you can try selecting different features, and then running the experiment again, to see if you get better results.

The various features are the weather conditions at the arrival and destination airports, departure and arrival times, the airline carrier, the day of month, and the day of the week.

[Step 4: Choose and Apply a Learning Algorithm]:#step-4-choose-and-apply-a-learning-algorithm

## <a name="anchor-4"></a> Choose and apply a learning algorithm.

**Model Training and Validation**

We created a model using the [**Two-Class Boosted Decision Tree**](https://msdn.microsoft.com/library/azure/e3c522f8-53d9-4829-8ea4-5c6a6b75330c) module and trained it using the training dataset.  To determine the optimal parameters, we connected the output port of **Two-Class Boosted Decision Tree**  to the [**Sweep Parameters**](https://msdn.microsoft.com/library/azure/038d91b6-c2f2-42a1-9215-1f2c20ed1b40) module.

![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight18b.PNG)

The model is optimized for the best AUC using 10-fold random parameter sweep.

![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight19.PNG)

For comparison, we created a model using the [**Two-Class Logistic Regression**](https://msdn.microsoft.com/library/azure/b0fd7660-eeed-43c5-9487-20d9cc79ed5d) module, and optimized it in the same manner.

The result of the experiment is a trained classification model that can be used to score new samples to make predictions. We used the validation set to generate scores from the trained models, and then used the [**Evaluate Model**](https://msdn.microsoft.com/library/azure/927d65ac-3b50-4694-9903-20f6c1672089) module to analyze and compare the quality of the models.

<a name="anchor-5"></a>
##Predict Using New Data
Now that we've trained the model, we can use it to score the other part of our data (the last month (October) records that were set aside for validation) and to see how well our model predicts and classifies new data.

Add the [**Score Model**](https://msdn.microsoft.com/library/azure/401b4f92-e724-4d5a-be81-d5b0ff9bdb33) module to the experiment canvas, and connect the left input port to the output of the **Train Model** module. Connect the right input port to the validation data (right port) of the [**Split**](https://msdn.microsoft.com/library/azure/70530644-c97a-4ab6-85f7-88bf30a8be5f) module.

After you run the experiment, you can view the output from the **Score Model** module by clicking the output port and selecting **Visualize**. The output includes the scored labels and the probabilities for the labels.

Finally, to test the quality of the results, add the [**Evaluate Model**](https://msdn.microsoft.com/library/azure/927d65ac-3b50-4694-9903-20f6c1672089) module to the experiment canvas, and connect the left input port to the output of the **Score Model** module. Note that there are two input ports for **Evaluate Model**, because the module can be used to compare two models. In this experiment, we compare the performance of the two different algorithms: the one created using **Two-Class Boosted Decision Tree** and the one created using **Two-Class Logistic Regression**.
Run the experiment and view the output of the **Evaluate Model** module, by clicking the output port and selecting **Visualize**.


##Results
The boosted decision tree model has AUC of 0.697 on the validation set, which is slightly better than the logistic regression model, with AUC of 0.675.
![screenshot_of_experiment](https://az712634.vo.msecnd.net/samplesimg/v1/4/flight20.PNG)


**Post-Processing**

To make the results easier to analyze, we used the _airportID_ field to join the dataset that contains the airport names and locations.




<!-- Images -->
[image1]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Anomaly_Detection_Credit_Risk/image1.PNG
[image2]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Anomaly_Detection_Credit_Risk/image2.PNG
[image3]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Anomaly_Detection_Credit_Risk/image3.PNG
[image4]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Anomaly_Detection_Credit_Risk/image4.PNG
[image5]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Anomaly_Detection_Credit_Risk/image5.PNG
[image6]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Anomaly_Detection_Credit_Risk/image6.PNG
[image7]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Anomaly_Detection_Credit_Risk/image7.PNG
[image8]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Anomaly_Detection_Credit_Risk/image8.PNG
