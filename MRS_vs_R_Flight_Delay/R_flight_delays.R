# Set up working directory.
# dataDir <- "<Where the 'Flight Delays Data.csv' and 'Weather Data.csv' are stored.>"

# Initial some variables.
inputFileFlight <- file.path(dataDir, "Flight Delays Data.csv")
inputFileWeather <- file.path(dataDir, "Weather Data.csv")


#### Step 1: Import Data.

# Import the flight data.
system.time(flight_r <- read.csv(file = inputFileFlight, na.strings = "NA", 
                                 stringsAsFactors = TRUE)
            )  # elapsed: 15.60 seconds

# Import the weather dataset.
# And eliminate some features due to redundance.
system.time(weather_r <- subset(read.csv(file = inputFileWeather, na.strings = "NA", 
                                         stringsAsFactors = TRUE),
                                select = -c(Year, Timezone, DryBulbFarenheit, DewPointFarenheit)
                                )
            )  # elapsed: 1.79 seconds


#### Step 2: Pre-process Data.

# Remove some columns that are possible target leakers from the flight data.
# And round down scheduled departure time to full hour.
xform_r <- function(df) {
  # Remove columns that are possible target leakers.
  varsToDrop <- c('DepDelay', 'DepDel15', 'ArrDelay', 'Cancelled', 'Year')
  df <- df[, !(names(df) %in% varsToDrop)]
  
  # Round down scheduled departure time to full hour.
  df$CRSDepTime <- floor(df$CRSDepTime/100)
  
  # Return the data frame.
  return(df)
}
system.time(flight_r <- xform_r(flight_r))  # elapsed: 0.03 seconds

# Rename some column names in the weather data to prepare it for merging.
xform2_r <- function(df) {
  # Create a new column 'DestAirportID' in weather data.
  df$DestAirportID <- df$AirportID
  
  # Rename 'AdjustedMonth', 'AdjustedDay', 'AirportID', 'AdjustedHour'.
  names(df)[match(c('AdjustedMonth', 'AdjustedDay', 'AirportID', 'AdjustedHour'),
                  names(df))] <- c('Month', 'DayofMonth', 'OriginAirportID', 'CRSDepTime')
  
  # Return the data frame.
  return(df)
}
system.time(weather_r <- xform2_r(weather_r))  # elapsed: <0.01 seconds

# Concatenate/Merge flight records and weather data.
# 1). Join flight records and weather data at origin of the flight (OriginAirportID).
mergeFunc <- function(df1, df2) {
  # Remove the "DestAirportID" column from the weather data before the merge.
  df2 <- subset(df2, select = -DestAirportID)
  
  # Merge the two data frames.
  dfOut <- merge(df1, df2, 
                 by = c('Month', 'DayofMonth', 'OriginAirportID', 'CRSDepTime'))
  
  # Return the data frame.
  return(dfOut)
}
system.time(originData_r <- mergeFunc(flight_r, weather_r)) # elapsed: 43.92 seconds

# 2). Join flight records and weather data using the destination of the flight (DestAirportID).
mergeFunc2 <- function(df1, df2) {
  # Remove the "OriginAirportID" column from the weather data before the merge.
  df2 <- subset(df2, select = -OriginAirportID)
  
  # Merge the two data frames.
  dfOut <- merge(df1, df2, 
                 by = c('Month', 'DayofMonth', 'DestAirportID', 'CRSDepTime'),
                 suffixes = c(".Origin", ".Destination"))
  
  # Return the data frame.
  return(dfOut)
}
system.time(destData_r <- mergeFunc2(originData_r, weather_r)) # elapsed: 37.53 seconds

# Normalize some numerical features and convert some features to be categorical.
# Features need to be normalized.
scaleVar <- c('Visibility.Origin', 'DryBulbCelsius.Origin', 'DewPointCelsius.Origin',
              'RelativeHumidity.Origin', 'WindSpeed.Origin', 'Altimeter.Origin',
              'Visibility.Destination', 'DryBulbCelsius.Destination', 'DewPointCelsius.Destination',
              'RelativeHumidity.Destination', 'WindSpeed.Destination', 'Altimeter.Destination')

# Features need to be converted to categorical.
cateVar <- c('OriginAirportID', 'DestAirportID')

xform3_r <- function(df) {
  # Normalization.
  df[, scaleVar] <- sapply(df[, scaleVar], FUN = function(x) {scale(x)})
  
  # Convert to categorical.
  df[, cateVar] <- sapply(df[, cateVar], FUN = function(x) {factor(x)})
  
  # Return the data frame.
  return(df)
}
system.time(finalData_r <- xform3_r(destData_r))  # elapsed: 39.98 seconds


#### Step 3: Prepare Training and Test Datasets.

# Randomly split 80% data as training set and the remaining 20% as test set.
set.seed(17)
system.time(sub <- sample(nrow(finalData_r), floor(nrow(finalData_r) * 0.8)))  # elapsed: 0.04 seconds
train <- finalData_r[sub, ]
test <- finalData_r[-sub, ]


#### Step 4A: Choose and apply a learning algorithm (Logistic Regression).

# Build the formula.
allvars <- names(finalData_r)
xvars <- allvars[allvars !='ArrDel15']
form <- as.formula(paste("ArrDel15", "~", paste(xvars, collapse = "+")))  

# Build a Logistic Regression model.
system.time(logitModel_r <- glm(form, data = train, family = "binomial"))  # elapsed: 229.32 seconds
summary(logitModel_r)


#### Step 5A: Predict over new data (Logistic Regression).

# Predict the probability on the test dataset.
system.time(predictLogit_r <- predict(logitModel_r, newdata = test, type = 'response'))  # elapsed: 1.72 seconds
testLogit <- cbind(test, data.frame(ArrDel15_Pred = predictLogit_r))

# Calculate Area Under the Curve (AUC).
auc <- function(outcome, prob){
  N <- length(prob)
  N_pos <- sum(outcome)
  df <- data.frame(out = outcome, prob = prob)
  df <- df[order(-df$prob),]
  df$above <- (1:N) - cumsum(df$out)
  return( 1- sum( df$above * df$out ) / (N_pos * (N-N_pos) ) )
}
auc(testLogit$ArrDel15, testLogit$ArrDel15_Pred)  # AUC = 0.70


#### Step 4B: Choose and apply a learning algorithm (Decision Tree).

# Build a decision tree model.
(if(!require("rpart")) install.packages("rpart"))
library(rpart)
system.time(dTree1_r <- rpart(form, data = train, method = 'class', 
                              control=rpart.control(minsplit = 20, minbucket = 1, cp = 0,
                                                    maxcompete = 0, maxSurrogate = 0, 
                                                    xval = 2, maxDepth = 10)))  # elapsed: 360.10 seconds

# Find the Best Value of cp for Pruning rxDTree Object.
treeCp_r <- dTree1_r$cptable[which.min(dTree1_r$cptable[,"xerror"]),"CP"]  # treeCp_r = 1.063143e-05

# Prune a decision tree created by rxDTree and return the smaller tree.
system.time(dTree2_r <- prune(dTree1_r, cp = treeCp_r))   # elapsed: 4.25 seconds


#### Step 5B: Predict over new data (Decision Tree).

# Predict the probability on the test dataset.
system.time(predictTree_r <- predict(dTree2_r, newdata = test, type = 'prob'))  # elapsed: 13.53 seconds
testDT <- cbind(test, data.frame(ArrDel15_Pred = predictTree_r[, 2]))

# Calculate Area Under the Curve (AUC).
auc(testDT$ArrDel15, testDT$ArrDel15_Pred)  # AUC = 0.74


