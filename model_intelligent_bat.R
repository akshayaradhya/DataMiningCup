
library(caret)
library(chron)
library(rpart)
library(ipred)
library(RWeka)
library(rJava)
library(partykit)
library(party)
library(RWekajars)
library(FSelector)
library(e1071)
library(Metrics)
library(rpart)

set.seed(42) 

training_data = read.csv("DMC_training_data_n9tGsjU.csv", sep=",", na.string="")
attach(training_data)

str(training_data)


nrow(training_data)
ncol(training_data)

# Show the first and last rows
head(training_data)
tail(training_data)

# Show columns with missing values
colSums(is.na(training_data))


test_data = read.csv("pub_KqMUzC8.csv", sep=",", na.string="")


date_format = "%d-%m-%Y"


# Factoring required columns and Parsing new data
training_data$portNumber <- as.factor(training_data$portNumber)
training_data$TYP1_COUNT <- as.factor(training_data$TYP1_COUNT)
training_data$ADDR_POSTALCODE <- as.factor(training_data$ADDR_POSTALCODE)
training_data$EI65_GEO_ID <- as.factor(training_data$EI65_GEO_ID)
training_data$status <- as.factor(training_data$status)
training_data$ADDR_STREET <- as.factor(training_data$ADDR_STREET)

training_data$day_timestamp <- weekdays(as.Date(as.character(training_data$LAST_MODIFIED,date_format)))
training_data$hour_timestamp <- as.numeric(format(as.POSIXct(c(as.character(training_data$LAST_MODIFIED))),"%H"))


training_data$day_timestamp <- as.factor(training_data$day_timestamp)
training_data$hour_timestamp <- as.factor(training_data$hour_timestamp)


test_data$portNumber <- as.factor(test_data$portNumber)
test_data$TYP1_COUNT <- as.factor(test_data$TYP1_COUNT)
test_data$ADDR_POSTALCODE <- as.factor(test_data$ADDR_POSTALCODE)
test_data$EI65_GEO_ID <- as.factor(test_data$EI65_GEO_ID)
test_data$status <- as.factor(test_data$status)
test_data$ADDR_STREET <- as.factor(test_data$ADDR_STREET)

test_data$day_timestamp <- weekdays(as.Date(as.character(test_data$LAST_MODIFIED,date_format)))
test_data$hour_timestamp <- as.numeric(format(as.POSIXct(c(as.character(test_data$LAST_MODIFIED))),"%H"))

test_data$day_timestamp <- as.factor(test_data$day_timestamp)
test_data$hour_timestamp <- as.factor(test_data$hour_timestamp)

#Dropping columns not holding significance
drops <- c("HOUSEHOLD_COUNT","ADDR_MUNICIPALITY","TimeStamp","ADDR_LATITUDE","ADDR_LONGITUDE","ADDR_REGION","PREFERRED_PARTNER","VALIDATION_LAST_MODIFIED")

drops_test <- c("HOUSEHOLD_COUNT","ADDR_MUNICIPALITY","TimeStamp","ADDR_LATITUDE","ADDR_LONGITUDE","ADDR_REGION","PREFERRED_PARTNER","VALIDATION_LAST_MODIFIED")

training_data <- training_data[ , !(names(training_data) %in% drops)]
test_data <- test_data[ , !(names(test_data) %in% drops_test)]

training_data$FREECHARGE <- as.character(training_data$FREECHARGE)
training_data[is.na(training_data)] <- "NO"
training_data$FREECHARGE <- as.factor(training_data$FREECHARGE)

test_data$FREECHARGE <- as.character(test_data$FREECHARGE)
test_data[is.na(test_data)] <- "NO"
test_data$FREECHARGE <- as.factor(test_data$FREECHARGE)

colSums(is.na(training_data))



m1 <- J48(training_data$status ~ ., data = training_data)

summary(m1)

fitCtrl = trainControl(method="repeatedcv", number=10, repeats=2)
weights_info_gain = information.gain(status ~ ., data=training_data)
weights_gain_ratio = gain.ratio(status ~ ., data=training_data)
most_important_attributes <- cutoff.k(weights_gain_ratio, 7)
formula_with_most_important_attributes <- as.simple.formula(most_important_attributes, "status")
model = train(formula_with_most_important_attributes, data=training_data, method="J48", trControl=fitCtrl, metric="Accuracy",  na.action=na.omit)
model
model$results

# Show decision tree
model$finalModel

# Show confusion matrix (in percent)
confusionMatrix(model)

######################################################
# 5. Predict Classes in Test Data
prediction_classes = predict.train(object=model, newdata=test_data, na.action=na.pass)
predictions = data.frame(id=test_data$ID, prediction=prediction_classes)
predictions


######################################################
# 6. Export the Predictions
write.csv(predictions, file="predictions_intelligent_bat_1.csv", row.names=FALSE)

