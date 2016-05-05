# Bike Rental Demand Estimation with Microsoft R Server #

Accurate demand forecasting empowers companies across various industries to perform effective production planning and resource allocation. Rented bicycle has become popular as a convenient and environmentally friendly transportation option. Accurate estimation of bike demand at different locations and different time will help bicycle-sharing systems better meet rental demand, allocate bikes based on estimated demand, and reduce storage cost. In this blog, we will walk through how to use Microsoft R Server (MRS) to build a regression model to predict bike rental demand. In the example below, we demonstrate an end-to-end machine learning solution development process in MRS, including data importing, data cleaning, feature engineering, parameter sweeping, and model training and evaluation.

## Data ##

The Bike Rental UCI dataset is used as the input raw data for this sample. This dataset is based on real-world data from the Capital Bikeshare company, which operates a bike rental network in Washington DC in the United States.

The dataset contains 17,379 rows and 17 columns, with each row representing the number of bike rentals within a specific hour of a day in the years 2011 or 2012. Weather conditions (such as temperature, humidity, and wind speed) are included in this raw feature set, and the dates are categorized as holiday vs. weekday etc.

The field to predict is **cnt**, which contains a count value ranging from 1 to 977, representing the number of bike rentals within a specific hour.

## Model Overview ##

In this example, we use historical bike rental counts as well as the weather condition data to predict the number of bike rentals within a specific hour in the future. We approach this problem as a regression problem, since the label column (number of rentals) contains continuous real numbers.

Along this line, we split the raw data into two parts - data records in year 2011 to learn the regression model and data records in year 2012 to score and evaluate the model. Specifically, we employ the Decision Forest Regression algorithm as the regression model and build two models on datasets of different feature sets. Finally, we evaluate their prediction performance. We will elaborate the details in the following subsections.

## Microsoft R Server ##

We build the models using the `RevoScaleR` library in MRS. The `RevoScaleR` library provides extremely fast statistical analysis on terabyte-class data sets without needing specialized hardware. `RevoScaleR`'s distributed computing capabilities can establish different computing context while remaining same `RevoScaleR` commands to manage and analyze data. A wide range of `rx` prefixed functions that include functionality for:

- Accessing external data sets (SAS, SPSS, ODBC, Teradata, and delimited and fixed format text) for analysis in R.
- Efficiently storing and retrieving data in a high-performance data file.
- Cleaning, exploring, and manipulating data.
- Fast, basic statistical analysis.
- Train and score advanced machine learning models.

## Running the Experiment ##

Overall, there are five major steps of building this example using Microsoft R Server:

- [Step 1: Import and Clean Data](#step-1)
- [Step 2: Feature Engineering](#step-2)
- [Step 3: Prepare Training, Test and Score Datasets](#step-3)
- [Step 4: Train the Model](#step-4)
- [Step 5: Test, Evaluate, and Compare the Model](#step-5)

### <a name="step-1"></a>Step 1: Import and Clean Data

Before importing the Bike Rental UCI dataset, we incorporate parallel computing on a local machine using the special compute context `RxLocalParallel` as follows:

```r
rxSetComputeContext(RxLocalParallel())
```

Since there are a small portion of missing records within the dataset, we use `rxDataStep()` to replace the missing records with the latest non-missing observations. `rxDataStep()` is good function to utilize for data manipulation. It transforms the input dataset chunk by chunk and saves the results to the output dataset.

```r
# Define the tranformation function for the rxDataStep.
xform <- function(dataList) {
  # Identify the features with missing values.
  featureNames <- c("weathersit", "temp", "atemp", "hum", "windspeed", "cnt")
  # Use "na.locf" function to carry forward last observation.
  dataList[featureNames] <- lapply(dataList[featureNames], zoo::na.locf)
  # Return the data list.
  return(dataList)
}

# Use rxDataStep to replace missings with the latest non-missing observations.
cleanXdf <- rxDataStep(inData = mergeXdf, outFile = outFileClean, overwrite = TRUE,
                       # Apply the "last observation carried forward" operation.
                       transformFunc = xform,
                       # Identify the features to apply the tranformation.
                       transformVars = c("weathersit", "temp", "atemp", "hum", "windspeed", "cnt"),
                       # Drop the "dteday" feature.
                       varsToDrop = "dteday")
```

### <a name="step-2"></a>Step 2: Feature Engineering

In addition to the original features in the raw data, we add number of bikes that were rented in each of the previous 12 hours as features to provide better predictive power. We write a `computeLagFeatures()` helper function to compute the 12 lag features. Then the `computeLagFeatures()` function is applied on the bike data via the `rxDataStep()` function. Note that rxDataStep() processes data in chunk by chunk and lag feature computation requires data from previous rows. In computLagFeatures(), we use the internal function `.rxSet()` to save the last n rows of a chunk to a variable **lagData**. When processing the next chunk, we use another internal function `.rxGet()` to retrieve the values stored in **lagData** and compute the lag features.

```r
# Add number of bikes that were rented in each of
# the previous 12 hours as 12 lag features.
computeLagFeatures <- function(dataList) {
  # Total number of lags that need to be added.
  numLags <- length(nLagsVector)
  # Lag feature names as lagN.
  varLagNameVector <- paste("cnt_", nLagsVector, "hour", sep="")

  # Set the value of an object "storeLagData" in the transform environment.
  if (!exists("storeLagData"))
  {
    lagData <- mapply(rep, dataList[[varName]][1], times = nLagsVector)
    names(lagData) <- varLagNameVector
    .rxSet("storeLagData", lagData)
  }

  if (!.rxIsTestChunk)
  {
    for (iL in 1:numLags)
    {
      # Number of rows in the current chunk.
      numRowsInChunk <- length(dataList[[varName]])
      nlags <- nLagsVector[iL]
      varLagName <- paste("cnt_", nlags, "hour", sep = "")
      # Retrieve lag data from the previous chunk.
      lagData <- .rxGet("storeLagData")
      # Concatenate lagData and the "cnt" feature.
      allData <- c(lagData[[varLagName]], dataList[[varName]])
      # Take the first N rows of allData, where N is
      # the total number of rows in the original dataList.
      dataList[[varLagName]] <- allData[1:numRowsInChunk]
      # Save last nlag rows as the new lagData to be used
      # to process in the next chunk.
      lagData[[varLagName]] <- tail(allData, nlags)
      .rxSet("storeLagData", lagData)
    }
  }
  return(dataList)
}

# Apply the "computeLagFeatures" on the bike data.
lagXdf <- rxDataStep(inData = cleanXdf, outFile = outFileLag,
                     transformFunc = computeLagFeatures,
                     transformObjects = list(varName = "cnt",
                                             nLagsVector = seq(12)),
                     transformVars = "cnt", overwrite = TRUE)
```

### <a name="step-3"></a>Step 3: Prepare Training, Test and Score Datasets

Before training the regression model, we first split data into two parts - data records in year 2011 to learn the regression model and data records in year 2012 to score and evaluate the model. In order to find the best combination of parameters for  the regression model, we further divide the year 2011 data into the training and test datasets - we randomly select 80% records from year 2011 data to train regression models with various combination of parameters, and use the residual 20% data to evaluate the models obtained and determine the optimal combination.

```r
# Split data by "yr" so that the training and test data contains records
# for the year 2011 and the score data contains records for 2012.
rxSplit(inData = lagXdf,
        outFilesBase = paste0(td, "/modelData"),
        splitByFactor = "yr",
        overwrite = TRUE,
        reportProgress = 0,
        verbose = 0)

# Point to the .xdf files for the training & test and score set.
trainTest <- RxXdfData(paste0(td, "/modelData.yr.0.xdf"))
score <- RxXdfData(paste0(td, "/modelData.yr.1.xdf"))

# Randomly split records for the year 2011 into training and test sets
# for sweeping parameters.
# 80% of data as training and 20% as test.
rxSplit(inData = trainTest,
        outFilesBase = paste0(td, "/sweepData"),
        outFileSuffixes = c("Train", "Test"),
        splitByFactor = "splitVar",
        overwrite = TRUE,
        transforms = list(splitVar = factor(sample(c("Train", "Test"),
                                                   size = .rxNumRows,
                                                   replace = TRUE,
                                                   prob = c(.80, .20)),
                                            levels = c("Train", "Test"))),
        rngSeed = 17,
        consoleOutput = TRUE)

# Point to the .xdf files for the training and test set.
train <- RxXdfData(paste0(td, "/sweepData.splitVar.Train.xdf"))
test <- RxXdfData(paste0(td, "/sweepData.splitVar.Test.xdf"))
```

### <a name="step-4"></a>Step 4: Sweep Parameter and Train the Model

In this step, we construct two training datasets based on the same raw input data, but with different sets of features:

- Set A = weather + holiday + weekday + weekend features for the predicted day
- Set B = number of bikes that were rented in each of the previous 12 hours, which captures very recent demand for the bikes

Then, two training datasets are built by combining the feature set as follows:
- Training set 1: feature set A only
- Training set 2: feature sets A+B

For each training dataset, a list of Decision Forest Regression models are trained with various combination of parameters. The test data is used to determine the optimal parameters.

Specifically, we first create a helper function to evaluate the performance of a model trained with a given combination of number of trees and maximum depth. We use _Root Mean Squared Error (RMSE)_ as the evaluation metric.

```r
# Define a function to train and test models with given parameters
# and then return Root Mean Squared Error (RMSE) as the performance metric.
TrainTestDForestfunction <- function(trainData, testData, form, numTrees, maxD)
{
  # Build decision forest regression models with given parameters.
  dForest <- rxDForest(form, data = trainData,
                       method = "anova",
                       maxDepth = maxD,
                       nTree = numTrees,
                       seed = 123)
  # Predict the the number of bike rental on the test data.
  rxPredict(dForest, data = testData,
            predVarNames = "cnt_Pred",
            residVarNames = "cnt_Resid",
            overwrite = TRUE,
            computeResiduals = TRUE)
  # Calcualte the RMSE.
  result <- rxSummary(~ cnt_Resid,
                      data = testData,
                      summaryStats = "Mean",
                      transforms = list(cnt_Resid = cnt_Resid^2)
  )$sDataFrame
  # Return lists of number of trees, maximum depth and RMSE.
  return(c(numTrees, maxD, sqrt(result[1,2])))
}
```

Next, we define the 9 combinations of parameters the model going to explore. We create a small number of combinations for demonstration purpose. A much larger number of parameter combinations are usually swept through in data science solution development.

```r
# Define a list of parameters to sweep through.
# To save time, we only sweep 9 combinations of number of trees and max tree depth.
numTreesToSweep <- rep(seq(20, 60, 20), times = 3)
maxDepthToSweep <- rep(seq(10, 30, 10), each = 3)
```

We create another helper function to sweep and select the optimal parameter combination. Under local parallel compute context, `rxExec()` executes multiple runs of model training and evaluation with different parameters in parallel, which significantly speeds up parameter sweeping. When used in a compute context with multiple nodes, e.g. high-performance computing clusters and Hadoop, `rxExec()` can be used to distribute a large number of tasks to the nodes and run the tasks in parallel.

```r
# Define a function to sweep and select the optimal parameter combination.
findOptimal <- function(DFfunction, train, test, form, nTreeArg, maxDepthArg) {
  # Sweep different combination of parameters.
  sweepResults <- rxExec(DFfunction, train, test, form, rxElemArg(nTreeArg), rxElemArg(maxDepthArg))
  # Sort the nested list by the third element (RMSE) in the list in ascending order.
  sortResults <- sweepResults[order(unlist(lapply(sweepResults, `[[`, 3)))]
  # Select the optimal parameter combination.
  nTreeOptimal <- sortResults[[1]][1]
  maxDepthOptimal <- sortResults[[1]][2]
  # Return the optimal values.
  return(c(nTreeOptimal, maxDepthOptimal))
}
```

We first find the optimal parameter combination on feature set A, and get the trained optimal regression model for set A.

```r
# Set A = weather + holiday + weekday + weekend features for the predicted day.
# Build a formula for the regression model and remove the "yr",
# which is used to split the training and test data.
newHourFeatures <- paste("cnt_", seq(12), "hour", sep = "")  # Define the hourly lags.
formA <- formula(train, depVars = "cnt", varsToDrop = c("splitVar", newHourFeatures))

# Find the optimal parameters for Set A.
optimalResultsA <- findOptimal(TrainTestDForestfunction,
                               train, test, formA,
                               numTreesToSweep,
                               maxDepthToSweep)

# Use the optimal parameters to fit a model for feature Set A.
nTreeOptimalA <- optimalResultsA[[1]]
maxDepthOptimalA <- optimalResultsA[[2]]
dForestA <- rxDForest(formA, data = train,
                      method = "anova",
                      maxDepth = maxDepthOptimalA,
                      nTree = nTreeOptimalA,
                      importance = TRUE, seed = 123)
```

Then we repeat the same process on feature set B.

```r
# Set B = number of bikes that were rented in each of the
# previous 12 hours, which captures very recent demand for the bikes.
formB <- formula(train, depVars = "cnt", varsToDrop = c("splitVar", "yr"))

# Find the optimal parameters for Set B.
optimalResultsB <- findOptimal(TrainTestDForestfunction,
                               train, test, formB,
                               numTreesToSweep,
                               maxDepthToSweep)

# Use the optimal parameters to fit a model for feature Set B.
nTreeOptimalB <- optimalResultsB[[1]]
maxDepthOptimalB <- optimalResultsB[[2]]
dForestB <- rxDForest(formB, data = train,
                      method = "anova",
                      maxDepth = maxDepthOptimalB,
                      nTree = nTreeOptimalB,
                      importance = TRUE, seed = 123)
```

Finally, we plot the dot charts of the variable importance and the out-of-bag error rates versus the number of trees for the two optimal decision forest models.

![](https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Bike_Rental_Estimation/image/1.png)

![](https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Bike_Rental_Estimation/image/2.png)

### <a name="step-5"></a>Step 5: Test, Evaluate, and Compare the Model

In this step, we use the `rxPredict()` function to predict the bike rental number on the score dataset, and compare the two regression models over three performance metrics - _Mean Absolute Error (MAE)_, _Root Mean Squared Error (RMSE)_, and _Relative Absolute Error (RAE)_.

```r
# Set A: Predict the probability on the test dataset.
rxPredict(dForestA, data = score,
          predVarNames = "cnt_Pred_A",
          residVarNames = "cnt_Resid_A",
          overwrite = TRUE, computeResiduals = TRUE)

# Set B: Predict the probability on the test dataset.
rxPredict(dForestB, data = score,
          predVarNames = "cnt_Pred_B",
          residVarNames = "cnt_Resid_B",
          overwrite = TRUE, computeResiduals = TRUE)

# Calculate three statistical metrics:
# Mean Absolute Error (MAE),
# Root Mean Squared Error (RMSE), and
# Relative Absolute Error (RAE).
sumResults <- rxSummary(~ cnt_Resid_A_abs + cnt_Resid_A_2 + cnt_rel_A +
                   cnt_Resid_B_abs + cnt_Resid_B_2 + cnt_rel_B,
                 data = score,
                 summaryStats = "Mean",
                 transforms = list(cnt_Resid_A_abs = abs(cnt_Resid_A),
                                   cnt_Resid_A_2 = cnt_Resid_A^2,
                                   cnt_rel_A = abs(cnt_Resid_A)/cnt,
                                   cnt_Resid_B_abs = abs(cnt_Resid_B),
                                   cnt_Resid_B_2 = cnt_Resid_B^2,
                                   cnt_rel_B = abs(cnt_Resid_B)/cnt)
)$sDataFrame

# Add row names.
features <- c("baseline: weather + holiday + weekday + weekend features for the predicted day",
              "baseline + previous 12 hours demand")

# List all metrics in a data frame.
metrics <- data.frame(Features = features,
                       MAE = c(sumResults[1, 2], sumResults[4, 2]),
                       RMSE = c(sqrt(sumResults[2, 2]), sqrt(sumResults[5, 2])),
                       RAE = c(sumResults[3, 2], sumResults[6, 2]))

# Review the metrics
metrics
```

Based on all three metrics listed below, the regression model built on feature set B outperforms the one built on feature set A. This result is not surprising, since from the variable importance chart we can see, the lag features play a critical part in the regression model. Adding this set of feature into feature set A definitely leads to better performance (Feature set B = Feature A + lag features).

Feature Set | MAE | RMSE | RAE
--- | --- | --- | ---
A | 100.8536 | 145.5507 | 0.9498727
B | 62.1749 | 103.4644 | 0.3799470
