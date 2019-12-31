# This R script is used to build a neural network model for classifying images of breast mass into 2 categories;
# The data used for the classification task is the breast cancer diagnostic data set
# Written by Adefunke Adeshina


##Section to load required packages in memory

library("neuralnet")               # Load the neuralnet package in memory
library ("NeuralNetTools")         # Load the NeuralNetTools package in memory
library('ggplot2')
library('ggthemes')
library('ggridges')
library('ggforce')
library('ggExtra')
library('grid')
library('gridExtra')


#**************************************************************************************
##Set working directory and read the data
# set the working directory
setwd("C:/Users/Jummmii/Documents/UMUC/DATA630/mod 5")
# Load the data into the data frame 'bcd'
bcd <-read.csv(file="wdbc.csv", head=TRUE, sep=",")

#************************************************************************************
## Section for preliminary data exploration
dim(bcd)
# A preview of the first 6 data rows
head(bcd)
# List of variable names
names(bcd)
# To view data structure
str(bcd)
# Descriptive statistics for all variables
summary(bcd)

#****************************************************************************************
## Section for Data preprocessing

# Convert diagnosis variable to numeric and create a new variable
bcd$diagnosis_new[bcd$diagnosis == "M"] <- 1
bcd$diagnosis_new[bcd$diagnosis == "B"] <- 0
# To view variable structure
str(bcd$diagnosis_new)

# Remove ID column
bcd$ID <- NULL

# Normalize the input variables
bcd[2:31]<-scale(bcd[2:31])
# Check if the data has missing values
colSums(is.na(bcd))

#**********************************************************************************************
## Section for exploratory analysis
# Distribution of diagnosis variable
ggplot(bcd) + 
  geom_bar(aes(diagnosis))

# Scatterplot of the variables values
pairs(bcd[,2:6])
pairs(bcd[,7:11])
pairs(bcd[,27:31])

# Distribution of diagnosis in relation to other features
ggplot(bcd) +
  geom_freqpoly(aes(x=concavity3, color=diagnosis), binwidth=1) +
  guides(fill=FALSE)

ggplot(bcd) +
  geom_freqpoly(aes(x=fractal, color=diagnosis), binwidth=1) +
  guides(fill=FALSE)

ggplot(bcd) +
  geom_freqpoly(aes(x=radius, color=diagnosis), binwidth=1) +
  guides(fill=FALSE)
ggplot(bcd) +
  geom_freqpoly(aes(x=texture, color=diagnosis), binwidth=1) +
  guides(fill=FALSE)

# New dataframe without diagnosis variable
new_bcd <- bcd
new_bcd$diagnosis <- NULL


#*********************************************************************************************
## Section to split the data into training and test set
set.seed(1234)
ind <- sample(2, nrow(new_bcd), replace = TRUE, prob = c(0.7, 0.3))
train.data <- new_bcd[ind == 1, ]  #70%
test.data <- new_bcd[ind == 2, ]  #30%

# Build model on training data
# Help page for the neuralnet method to see default settings
?neuralnet

# Specify dependent and independent variables
myFormula <- diagnosis_new~radius + texture + perimeter + area + smoothness + 
  compactness + concavity + concave + symmetry + fractal + radius2 + texture2 + perimeter2 + 
  area2 + smoothness2 + compactness2 + concavity2 + concave2 + symmetry2 + fractal2 + radius3 + 
  texture3 + perimeter3 + area3 + smoothness3 + compactness3 + concavity3 + concave3 + 
  symmetry3 + fractal3

# Build model with neuralnet function and store result in variable nn
nn <- neuralnet(myFormula, data=train.data, hidden=c(3), err.fct = "ce", linear.output = FALSE)


# Check the available network properties
names(nn)
# The final weights, number of steps, error, and threshold
nn$result.matrix  
# Neural network, nn visualization
plot(nn)
plotnet(nn)

# Prediction with training data
mypredict<-compute(nn, nn$covariate)$net.result
mypredict<-apply(mypredict, c(1), round)            # Round the predicted probabilities

# Compute confusion matrix with training data using model nn
table(mypredict, train.data$diagnosis_new, dnn =c("Predicted", "Actual"))
mean(mypredict==train.data$diagnosis_new)

# Evaluate model nn on test data
testPred <- compute(nn, test.data[, 1:30])$net.result
testPred <- apply(testPred, c(1), round)
# Compute confusion matrix on test data
table(testPred, test.data$diagnosis_new, dnn =c("Predicted", "Actual"))
mean(testPred==test.data$diagnosis_new)

# Relative importance of input variables
garson(nn)

# Preview of first 10 predicted values and actual values
pred <- compute(nn, test.data[, 1:30])$net.result
result <- data.frame(actual = test.data$diagnosis_new[1:10], predicted = pred[1:10])
result

#****************************************************************************************

# Build another model with 6 hidden nodes and store result in variable nn2
nn2 <- neuralnet(myFormula, data=train.data, hidden=c(6), err.fct = "ce", linear.output = FALSE)

# The final weights, number of steps, error, and threshold for nn2
nn2$result.matrix

# Neural network visualization for nn2
plot(nn2)

# Confusion matrix for training data with model nn2
mypredict2<-compute(nn2, nn2$covariate)$net.result
mypredict2<-apply(mypredict2, c(1), round)            # Round the predicted probabilities
                            
table(mypredict2, train.data$diagnosis_new, dnn =c("Predicted", "Actual"))
mean(mypredict2==train.data$diagnosis_new)

# Evaluate model nn2 on test data
testPred2 <- compute(nn2, test.data[, 1:30])$net.result
testPred2 <- apply(testPred2, c(1), round)

table(testPred2, test.data$diagnosis_new, dnn =c("Predicted", "Actual"))
mean(testPred2==test.data$diagnosis_new)

#***********************************************************************************************
## Build neural network nn3 with training data

# Build model with neuralnet function and store result in variable nn3
nn3 <- neuralnet(myFormula, data=train.data, hidden=c(4,2), err.fct = "ce", linear.output = FALSE)

# The final weights, number of steps, error, and threshold
nn3$result.matrix  
# Neural network visualization
plot(nn3)

# Prediction with training data
mypredict3<-compute(nn3, nn3$covariate)$net.result
mypredict3<-apply(mypredict3, c(1), round)            # Round the predicted probabilities

# Compute confusion matrix with training data using model nn3
table(mypredict3, train.data$diagnosis_new, dnn =c("Predicted", "Actual"))
mean(mypredict3==train.data$diagnosis_new)

# Evaluate model nn3 on test data
testPred3 <- compute(nn3, test.data[, 1:30])$net.result
testPred3 <- apply(testPred3, c(1), round)
# Compute confusion matrix on test data
table(testPred3, test.data$diagnosis_new, dnn =c("Predicted", "Actual"))
mean(testPred3==test.data$diagnosis_new)


# End of script
