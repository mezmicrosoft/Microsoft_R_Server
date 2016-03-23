# Set up working directory.
dataDir <- "<Where the 'Flight Delays Data.csv' and 'Weather Data.csv' are stored.>"

# Initial some variables.
inputFileFlight <- file.path(dataDir, "Flight Delays Data.csv")
inputFileWeather <- file.path(dataDir, "Weather Data.csv")
outFileFlight <- 'flight.xdf'
outFileFlight2 <- 'flight2.xdf'
outFileWeather <- 'weather.xdf'
outFileWeather2 <- 'weather2.xdf'
outFileOrigin <- 'originData.xdf'
outFileDest <- 'DestData.xdf'
outFileFinal <- 'finalData.xdf'


#### Step 1: Import Data.

# Import the flight data.
(if (file.exists(file.path(getwd(), outFileFlight))) {file.remove(outFileFlight)})
flight <- rxImport(inData = inputFileFlight, outFile = outFileFlight, 
                  missingValueString = "M", stringsAsFactors = TRUE)

# Examine the imported flight data.
dim(flight)  # 2719418*14

# Get .xdf File Information.
rxGetInfo(flight)

# Get variable information of flight data.
rxGetVarInfo(flight)

# Review the first 6 rows of flight data.
head(flight)

# Summary the flight data.
rxSummary(~., data = flight, blocksPerRead = 2)

# Import the weather dataset.
# And eliminate some features due to redundance.
weather <- rxImport(inData = inputFileWeather, outFile = outFileWeather, 
                    missingValueString = "M", stringsAsFactors = TRUE,
                    varsToDrop = c('Year', 'Timezone', 'DryBulbFarenheit', 'DewPointFarenheit'),
                    overwrite=TRUE)

# Examine the imported weather data.
nrow(weather)  # 404914
ncol(weather)  # 10

# Summary the weather data.
summary(weather)



#### Step 2: Pre-process Data.

# Remove columns that are possible target leakers from the flight data. 
varsToDrop <- c('DepDelay', 'DepDel15', 'ArrDelay', 'Cancelled', 'Year')

# Round down scheduled departure time to full hour.
xform <- function(dataList) {
  # Create a new continuous variable from an existing continuous variables:
  # round down CSRDepTime column to the nearest hour.
  dataList$CRSDepTime <- sapply(dataList$CRSDepTime, 
                                FUN = function(x) {floor(x/100)})
  
  # Return the adapted variable list.
  return(dataList)
}
flight <- rxDataStep(inData = flight, 
                     outFile = outFileFlight2,
                     varsToDrop = varsToDrop,
                     transformFunc = xform, 
                     transformVars = 'CRSDepTime',
                     overwrite=TRUE
                     )

# Rename some column names in the weather data to prepare it for merging.
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

# Concatenate/Merge flight records and weather data.
# 1). Join flight records and weather data at origin of the flight (OriginAirportID).
originData <- rxMerge(inData1 = flight, inData2 = weather, outFile = outFileOrigin,
                      type = 'inner', autoSort = TRUE, decreasing = FALSE,
                      matchVars = c('Month', 'DayofMonth', 'OriginAirportID', 'CRSDepTime'), 
                      varsToDrop2 = 'DestAirportID',
                      overwrite=TRUE
                      )

# 2). Join flight records and weather data using the destination of the flight (DestAirportID).
destData <- rxMerge(inData1 = originData, inData2 = weather, outFile = outFileDest,
                    type = 'inner', autoSort = TRUE, decreasing = FALSE,
                    matchVars = c('Month', 'DayofMonth', 'DestAirportID', 'CRSDepTime'), 
                    varsToDrop2 = c('OriginAirportID'),
                    duplicateVarExt = c("Origin", "Destination"),
                    overwrite=TRUE
                    )

# Normalize some numerical features and convert some features to be categorical.
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



#### Step 3: Prepare Training and Test Datasets.

# Randomly split 80% data as training set and the remaining 20% as test set.
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

# Duplicate the test file for two models.
file.rename('finalData.splitVar.Test.xdf', 'finalData.splitVar.Test.logit.xdf')
file.copy('finalData.splitVar.Test.logit.xdf', 'finalData.splitVar.Test.tree.xdf')



#### Step 4A: Choose and apply a learning algorithm (Logistic Regression).

# Build the formula.
allvars <- names(finalData)
xvars <- allvars[allvars !='ArrDel15']
form <- as.formula(paste("ArrDel15", "~", paste(xvars, collapse = "+")))  

# Build a Logistic Regression model.
logitModel <- rxLogit(form, data = 'finalData.splitVar.Train.xdf')
summary(logitModel)



#### Step 5A: Predict over new data (Logistic Regression).

# Predict the probability on the test dataset.
predictLogit <- rxPredict(logitModel, data = 'finalData.splitVar.Test.logit.xdf', 
                          type = 'response', overwrite = TRUE)

# Show the first 5 rows of the prediction results.
rxGetInfo(predictLogit, getVarInfo = TRUE, numRows = 5)  # ArrDel15_Pred

# Set 0.5 as the threshold.
testDF <- rxImport('finalData.splitVar.Test.logit.xdf')
testDF$ArrDel15_Class[which(testDF$ArrDel15_Pred < 0.5)] <- 0
testDF$ArrDel15_Class[which(testDF$ArrDel15_Pred >= 0.5)] <- 1

# Plot ROC Curve.
rxRocCurve( "ArrDel15", "ArrDel15_Pred", predictLogit)

# Calculate Area Under the Curve (AUC).
rxAuc(rxRoc("ArrDel15", "ArrDel15_Pred", predictLogit))

# Compute Confusion matrix.
xtab <- table(testDF$ArrDel15_Class, testDF$ArrDel15)
(if(!require("e1071")) install.packages("e1071"))
(if(!require("caret")) install.packages("caret"))
library(e1071)
library(caret) 
confusionMatrix(xtab, positive = '1')



#### Step 4B: Choose and apply a learning algorithm (Decision Tree).

# Build a decision tree model.
dTree1 <- rxDTree(form, data = 'finalData.splitVar.Train.xdf')

# Find the Best Value of cp for Pruning rxDTree Object.
treeCp <- rxDTreeBestCp(dTree1)

# Prune a decision tree created by rxDTree and return the smaller tree.
dTree2 <- prune.rxDTree(dTree1, cp = treeCp)



#### Step 5B: Predict over new data (Decision Tree).

# Predict the probability on the test dataset.
predictTree <- rxPredict(dTree2, data = 'finalData.splitVar.Test.tree.xdf', 
                         overwrite = TRUE)

# Set 0.5 as the threshold.
testDF2 <- rxImport('finalData.splitVar.Test.tree.xdf')
testDF2$ArrDel15_Class[which(testDF2$ArrDel15_Pred < 0.5)] <- 0
testDF2$ArrDel15_Class[which(testDF2$ArrDel15_Pred >= 0.5)] <- 1

# Plot ROC Curve.
rxRocCurve( "ArrDel15", "ArrDel15_Pred", predictTree)

# Calculate Area Under the Curve (AUC).
rxAuc(rxRoc("ArrDel15", "ArrDel15_Pred", predictTree))

# Compute Confusion matrix.
xtab2 <- table(testDF2$ArrDel15_Class, testDF2$ArrDel15)
confusionMatrix(xtab2, positive = '1')


