## Regression: Demand estimation with Microsoft R Server

# This example demonstrates the how to solve a real world problem:
# predicting bike rental demand, using Microsoft R Server. 

# The dataset contains 17,379 rows and 17 columns, 
# each row representing the number of bike rentals within 
# a specific hour of a day in the years 2011 or 2012. 
# Weather conditions (such as temperature, humidity, 
# and wind speed) were included in this raw feature set, 
# and the dates were categorized as holiday vs. 
# weekday etc.

# The field to predict is "cnt", which contain a count value 
# ranging from 1 to 977, representing the number
# of bike rentals within a specific hour.

# We built two models using the same algorithm, 
# but with two different training datasets. The two training 
# datasets that we constructed were all based 
# on the same raw input data, but we added different additional 
# features to each training set.

# Set A = weather + holiday + weekday + weekend features 
# for the predicted day
# Set B = number of bikes that were rented in each of the 
# previous 12 hours, which captures very recent demand for the bikes.

# The two training datasets were built by combining the feature set 
# as follows:
# Training set 1: feature set A only
# Training set 2: feature sets A+B

# The following scripts include five basic steps of building 
# this example using Microsoft R Server.
# This execution might require more than two minutes.


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


### Step 3: Prepare Training and Test Datasets

# Split data by "yr" so that the training data contains records 
# for the year 2011 and the test data contains records for 2012.
rxSplit(inData = lagXdf,
        outFilesBase = paste0(td, "/modelData"),
        splitByFactor = "yr",
        overwrite = TRUE,
        reportProgress = 0,
        verbose = 0)

# Point to the .xdf files for the training and test set.
train <- RxXdfData(paste0(td, "/modelData.yr.0.xdf"))
test <- RxXdfData(paste0(td, "/modelData.yr.1.xdf"))


### Step 4: Choose and apply a learning algorithm (Decision Forest Regression)

# Define the hourly lags. 
newHourFeatures <- paste("cnt_", seq(12), "hour", sep = "")

# Set A = weather + holiday + weekday + weekend features for the predicted day.
# Build a formula for the regression model and remove the "yr", 
# which is used to split the training and test data.
formA <- formula(train, depVars = "cnt", varsToDrop = c("RowNum", "yr", newHourFeatures))

# Fit Decision Forest Regression model.
dForestA <- rxDForest(formA, data = train,
                      method = "anova", maxDepth = 10, nTree = 20,
                      importance = TRUE, seed = 123)

# Set B = number of bikes that were rented in each of the 
# previous 12 hours, which captures very recent demand for the bikes.
formB <- formula(train, depVars = "cnt", varsToDrop = c("RowNum", "yr"))

# Fit Decision Forest Regression model.
dForestB <- rxDForest(formB, data = train,
                      method = "anova", maxDepth = 10, nTree = 20,
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
rxPredict(dForestA, data = test, 
          predVarNames = "cnt_Pred_A",
          residVarNames = "cnt_Resid_A",
          overwrite = TRUE, computeResiduals = TRUE)

# Set B: Predict the probability on the test dataset.
rxPredict(dForestB, data = test, 
          predVarNames = "cnt_Pred_B",
          residVarNames = "cnt_Resid_B",
          overwrite = TRUE, computeResiduals = TRUE)

# Calculate three statistical measures: 
# Mean Absolute Error (MAE), 
# Root Mean Squared Error (RMSE), and 
# Relative Absolute Error (RAE).
sum <- rxSummary(~ cnt_Resid_A_abs + cnt_Resid_A_2 + cnt_rel_A +
                   cnt_Resid_B_abs + cnt_Resid_B_2 + cnt_rel_B,
                 data = test, summaryStats = "Mean", 
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

# List all measures in a data frame.
measures <- data.frame(Features = features,
                       MAE = c(sum[1, 2], sum[4, 2]),
                       RMSE = c(sqrt(sum[2, 2]), sqrt(sum[5, 2])),
                       RAE = c(sum[3, 2], sum[6, 2]))

# Review the measures.
measures

################################################################################################
##Sweep parameters to find the best hyperparameters for Decision Forest
################################################################################################
# Split training data into train and validation for sweeping parameters
rxSplit(inData = train,
        outFilesBase = paste0(td, "/sweepData"),
        outFileSuffixes = c("Train", "Validation"),
        splitByFactor = "splitVar",
        overwrite = TRUE,
        transforms = list(
          splitVar = factor(sample(c("Train", "Validation"),
                                   size = .rxNumRows,
                                   replace = TRUE,
                                   prob = c(.80, .20)),
                            levels = c("Train", "Validation"))),
        rngSeed = 17,
        consoleOutput = TRUE)

sweepTrain <- RxXdfData(paste0(td, "/sweepData.splitVar.Train.xdf"))
sweepValidation <- RxXdfData(paste0(td, "/sweepData.splitVar.Validation.xdf"))


#list of parameters to sweep through. To save time, we only sweep 9 combinations of number of trees and max tree depth
numTreesToSweep <- rep(seq(20,60,20),3)
maxDepthToSweep <- rep(seq(10,30,10),each=3)

#switch to local parallel compute context to sweep through parameters in parallel
rxSetComputeContext(RxLocalParallel())

#train and test model with given parameter, return RMSE
TrainTestDForestfunction <- function(trainData, testData,form,numTrees,maxD)
{
  dForest <- rxDForest(form, data = trainData,
                       method = "anova", maxDepth = maxD, nTree = numTrees,seed = 123)
  rxPredict(dForest, data = testData, 
            predVarNames = "cnt_Pred",
            residVarNames = "cnt_Resid",
            overwrite = TRUE, computeResiduals = TRUE)
  result <- rxSummary(~ cnt_Resid, 
                      data = testData, summaryStats = "Mean", 
                      transforms = list(cnt_Resid = cnt_Resid^2)
  )$sDataFrame
  
  return(c(numTrees,maxD,sqrt(result[1,2])))
  
}


#Sweep and select the optimal parameters for feature set A
sweepParamsResults_A <- rxExec(TrainTestDForestfunction, sweepTrain, sweepValidation,formA,rxElemArg(numTreesToSweep),rxElemArg(maxDepthToSweep))
sweepParamsResults_A <- t(data.frame(sweepParamsResults_A))
colnames(sweepParamsResults_A) <- c('numTrees','maxDepth','RMSE')
rownames(sweepParamsResults_A) <- seq(1,nrow(sweepParamsResults_A),1)
minRMSE_index_A = which.min(sweepParamsResults_A[,"RMSE"])[[1]]
numTrees_optimal_A = sweepParamsResults_A[minRMSE_index_A,"numTrees"]
maxDepth_optimal_A = sweepParamsResults_A[minRMSE_index_A,"maxDepth"]

#Sweep and select the optimal parameters for feature set B
sweepParamsResults_B <- rxExec(TrainTestDForestfunction, sweepTrain, sweepValidation,formB,rxElemArg(numTreesToSweep),rxElemArg(maxDepthToSweep))
sweepParamsResults_B <- t(data.frame(sweepParamsResults_B))
colnames(sweepParamsResults_B) <- c('numTrees','maxDepth','RMSE')
rownames(sweepParamsResults_B) <- seq(1,nrow(sweepParamsResults_B),1)
minRMSE_index_B = which.min(sweepParamsResults_B[,"RMSE"])[[1]]
numTrees_optimal_B = sweepParamsResults_B[minRMSE_index_B,"numTrees"]
maxDepth_optimal_B = sweepParamsResults_B[minRMSE_index_B,"maxDepth"]






