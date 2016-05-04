## Bike Rental Demand Estimation with Microsoft R Server

# The following scripts include five basic steps of building 
# this example using Microsoft R Server.


### Step 0: Get Started

# Check whether Microsoft R Server (RRE 8.0) is installed.
if (!require("RevoScaleR")) {
  cat("RevoScaleR package does not seem to exist. 
      \nThis means that the functions starting with 'rx' will not run. 
      \nIf you have Microsoft R Server installed, please switch the R engine.
      \nFor example, in R Tools for Visual Studio: 
      \nR Tools -> Options -> R Engine. 
      \nIf Microsoft R Server is not installed, you can download it from: 
      \nhttps://www.microsoft.com/en-us/server-cloud/products/r-server/
      \n")
  quit()
}

# Install the "zoo" package if it's not already installed.
(if (!require("zoo", quietly = TRUE)) install.packages("zoo"))

# Load package.
library("zoo", quietly = TRUE)

# Initialize some variables.
github <- "https://raw.githubusercontent.com/mezmicrosoft/Microsoft_R_Server/master/Bike_Rental_Estimation/data/"
inputeFileData <- paste0(github, "date_info.csv")
inputFileBike <- paste0(github, "Bike_Rental_UCI_Dataset.csv")

# Create a temporary directory to store the intermediate .xdf files.
td <- tempdir()
outFileDate <- paste0(td, "/date.xdf")
outFileBike <- paste0(td, "/bike.xdf")
outFileMerge <- paste0(td, "/merge.xdf")
outFileClean <- paste0(td, "/clean.xdf")
outFileLag <- paste0(td, "/lagData.xdf")


### Step 1: Import and Clean Data

# Import date information (dteday, season, yr, mnth, hr, holiday, weekday, workingday)
dateXdf <- rxImport(inData = inputeFileData,
                    outFile = outFileDate, overwrite = TRUE,
                    missingValueString = "M", stringsAsFactors = FALSE,
                    # Define categorical features.
                    colInfo = list(season = list(type = "factor"),
                                   yr = list(type = "factor"),
                                   mnth = list(type = "factor"),
                                   hr = list(type = "factor"),
                                   holiday = list(type = "factor"),
                                   weekday = list(type = "factor"),
                                   workingday = list(type = "factor")),
                    # Convert "dteday" feature from strings to dates.
                    # Extract "day" from the "dteday" feature.
                    transforms = list(dteday = as.Date(dteday, format="%m/%d/%Y"),
                                      day = as.numeric(format(dteday, "%d"))))

# Import bike rental data.
bikeXdf <- rxImport(inData = inputFileBike,
                    outFile = outFileBike, overwrite = TRUE,
                    missingValueString = "M", stringsAsFactors = FALSE,
                    # Drop some non-useful features.
                    varsToDrop = c("instant", 
                                   "casual", 
                                   "registered"),
                    # Define categorical features.
                    colInfo = list(season = list(type = "factor"),
                                   yr = list(type = "factor"),
                                   mnth = list(type = "factor"),
                                   hr = list(type = "factor"),
                                   holiday = list(type = "factor"),
                                   weekday = list(type = "factor"),
                                   workingday = list(type = "factor"),
                                   weathersit = list(type = "factor")),
                    # Convert "dteday" feature from strings to dates.
                    transforms = list(dteday = as.Date(dteday, format="%m/%d/%Y")))

# Left outer join date information and bike rental data.
mergeXdf <- rxMerge(inData1 = dateXdf, inData2 = bikeXdf, outFile = outFileMerge,
                    type = "left", autoSort = TRUE, decreasing = FALSE, 
                    # Joining key: "dteday", "hr"
                    matchVars = c("dteday", "hr"),
                    # Drop some duplicate features in the right table.
                    varsToDrop2 = c("season", "yr", "mnth", "holiday", "weekday", "workingday"),
                    overwrite = TRUE)

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


### Step 2: Feature Engineering

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


### Step 3: Prepare Training, Test and Score Datasets

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


### Step 4: Choose and apply Decision Forest Regression models
###         Sweep parameters to find the best hyperparameters

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

# Define a list of parameters to sweep through. 
# To save time, we only sweep 9 combinations of number of trees and max tree depth.
numTreesToSweep <- rep(seq(20, 60, 20), times = 3)
maxDepthToSweep <- rep(seq(10, 30, 10), each = 3)

# Switch to local parallel compute context to sweep through parameters in parallel.
rxSetComputeContext(RxLocalParallel())

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

# Plot two dotchart of the variable importance as measured by 
# the two decision forest models.
par(mfrow = c(1, 2))
rxVarImpPlot(dForestA, main = "Variable Importance of Set A")
rxVarImpPlot(dForestB, main = "Variable Importance of Set B")

# Plot Out-of-bag error rate comparing to the number of trees build 
# in a decision forest model.
plot(dForestA, main = "OOB Error Rate vs Number of Trees: Set A")
plot(dForestB, main = "OOB Error Rate vs Number of Trees: Set B")
par(mfrow = c(1, 1))


### Step 5: Predict over new data

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
