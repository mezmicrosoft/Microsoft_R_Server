# Set up working directory.
# dataDir <- "<Where the 'Flight Delays Data.csv' and 'Weather Data.csv' are stored.>"

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

# Turn off the progress reported in MRS.
rxOptions(reportProgress = 0)


#### Step 1: Import Data.

(if (file.exists(file.path(getwd(), outFileFlight))) {file.remove(outFileFlight)})

# Import the flight data.
system.time(flight_mrs <- rxImport(inData = inputFileFlight, outFile = outFileFlight, 
                                   missingValueString = "M", stringsAsFactors = TRUE)
            )  # elapsed: 10.86 seconds

# Import the weather dataset and eliminate some features due to redundance.
system.time(weather_mrs <- rxImport(inData = inputFileWeather, outFile = outFileWeather, 
                                    missingValueString = "M", stringsAsFactors = TRUE,
                                    varsToDrop = c('Year', 'Timezone', 'DryBulbFarenheit', 'DewPointFarenheit'),
                                    overwrite=TRUE)
            )  # elapsed: 2.33 seconds


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
system.time(flight_mrs <- rxDataStep(inData = flight_mrs, 
                                   outFile = outFileFlight2,
                                   varsToDrop = varsToDrop,
                                   transformFunc = xform, 
                                   transformVars = 'CRSDepTime',
                                   overwrite=TRUE
                                   )
            )  # elapsed: 4.75 seconds



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
system.time(weather_mrs <- rxDataStep(inData = weather_mrs,
                                      outFile = outFileWeather2, 
                                      transformFunc = xform2, 
                                      transformVars = c('AdjustedMonth', 'AdjustedDay', 'AirportID', 'AdjustedHour'),
                                      overwrite=TRUE
                                      )
            )  # elapsed: 0.34 seconds

# Concatenate/Merge flight records and weather data.
# 1). Join flight records and weather data at origin of the flight (OriginAirportID).
system.time(originData_mrs <- rxMerge(inData1 = flight_mrs, inData2 = weather_mrs, outFile = outFileOrigin,
                                      type = 'inner', autoSort = TRUE, decreasing = FALSE,
                                      matchVars = c('Month', 'DayofMonth', 'OriginAirportID', 'CRSDepTime'), 
                                      varsToDrop2 = 'DestAirportID',
                                      overwrite=TRUE
                                      )
            )  # elapsed: 28.44 seconds

# 2). Join flight records and weather data using the destination of the flight (DestAirportID).
system.time(destData_mrs <- rxMerge(inData1 = originData_mrs, inData2 = weather_mrs, outFile = outFileDest,
                                    type = 'inner', autoSort = TRUE, decreasing = FALSE,
                                    matchVars = c('Month', 'DayofMonth', 'DestAirportID', 'CRSDepTime'), 
                                    varsToDrop2 = c('OriginAirportID'),
                                    duplicateVarExt = c("Origin", "Destination"),
                                    overwrite=TRUE
                                    )
            )  # elapsed: 36.25 seconds

# Normalize some numerical features and convert some features to be categorical.
system.time(finalData_mrs <- rxDataStep(inData = destData_mrs, outFile = outFileFinal,
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
            )  # elapsed: 10.39 seconds



#### Step 3: Prepare Training and Test Datasets.

# Randomly split 80% data as training set and the remaining 20% as test set.
system.time(rxExec(rxSplit, inData = finalData_mrs,
                   outFilesBase="finalData",
                   outFileSuffixes=c("Train", "Test"),
                   splitByFactor="splitVar",
                   overwrite=TRUE,
                   transforms=list(splitVar = factor(sample(c("Train", "Test"), size=.rxNumRows, replace=TRUE, prob=c(.80, .20)),
                                   levels= c("Train", "Test"))), 
                   rngSeed=17, 
                   consoleOutput=TRUE
                   )
            )  # elapsed: 9.19 seconds

# Duplicate the test file for two models.
file.rename('finalData.splitVar.Test.xdf', 'finalData.splitVar.Test.logit.xdf')
file.copy('finalData.splitVar.Test.logit.xdf', 'finalData.splitVar.Test.tree.xdf')



#### Step 4A: Choose and apply a learning algorithm (Logistic Regression).

# Build the formula.
allvars <- names(finalData_mrs)
xvars <- allvars[allvars !='ArrDel15']
form <- as.formula(paste("ArrDel15", "~", paste(xvars, collapse = "+")))  

# Build a Logistic Regression model.
system.time(logitModel_mrs <- rxLogit(form, data = 'finalData.splitVar.Train.xdf'))  # elapsed: 5.50 seconds
summary(logitModel_mrs)



#### Step 5A: Predict over new data (Logistic Regression).

# Predict the probability on the test dataset.
system.time(predictLogit_mrs <- rxPredict(logitModel_mrs, data = 'finalData.splitVar.Test.logit.xdf', 
                                          type = 'response', overwrite = TRUE)
            ) # elapsed: 0.56 seconds

# Calculate Area Under the Curve (AUC).
rxAuc(rxRoc("ArrDel15", "ArrDel15_Pred", predictLogit_mrs))  # AUC = 0.70


#### Step 4B: Choose and apply a learning algorithm (Decision Tree).

# Build a decision tree model.
system.time(dTree1_mrs <- rxDTree(form, data = 'finalData.splitVar.Train.xdf'))  # elapsed: 151.69 seconds

# Find the Best Value of cp for Pruning rxDTree Object.
treeCp_mrs <- rxDTreeBestCp(dTree1_mrs)  # treeCp_mrs = 2.156921e-05

# Prune a decision tree created by rxDTree and return the smaller tree.
system.time(dTree2_mrs <- prune.rxDTree(dTree1_mrs, cp = treeCp_mrs))  # elapsed: 0.03 seconds



#### Step 5B: Predict over new data (Decision Tree).

# Predict the probability on the test dataset.
system.time(predictTree_mrs <- rxPredict(dTree2_mrs, data = 'finalData.splitVar.Test.tree.xdf', 
                                         overwrite = TRUE)
            )  # elapsed: 1.16 seconds

# Calculate Area Under the Curve (AUC).
rxAuc(rxRoc("ArrDel15", "ArrDel15_Pred", predictTree_mrs))  # AUC = 0.73

