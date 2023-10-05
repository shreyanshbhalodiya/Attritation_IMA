

install.packages("psych")

install.packages("glmnet")

install.packages("corrplot")

install.packages("RColorBrewer")

install.packages("Metrics")

install.packages("xgboost")

library(xgboost)

install.packages("caret")

library(psych)
library(Metrics)
library(psych)
#  library(glmnet)
library(Metrics)

# Final Project: Initial Analysis Report

# Exploratory Data Analysis

data <- read.csv("attrition.csv", header=TRUE,sep=",")

table(data$Attrition)

df_0 <- subset(data, Attrition == 0)
df_1 <- subset(data, Attrition == 1)
table(df_1$Attrition)

table(df_0$Attrition)
df_0_sam <- df_0[sample(nrow(df_0), 900), ]
table(df_0_sam$Attrition)

data <- rbind.data.frame(df_1,df_0_sam)
table(data$Attrition)

library(psych)

head(data)
summary(data)
str(data)
describe(data)

which(is.na(data))
# No missing values in the dataset

table(data$Attrition)
# Attriton variable is imbalanced with 1233 0's and 237 1's

hist(data$Age)
hist(data$TotalWorkingYears)
hist(data$MonthlyIncome)

data$Gender <- as.factor(data$Gender)

library(tidyverse)

data%>%
  ggplot(aes(x=MonthlyIncome, fill=Gender, color=Gender))+
  geom_density(alpha=0.4)+
  theme_bw()+
  labs(title = "DENSITY GRAPH OF INCOME BY GENDER")

library(corrplot)
library(RColorBrewer)

#creating subset with continous variables

data_cont <- subset(data, select=c(Age,MonthlyIncome,TotalWorkingYears,
                                   YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,
                                   YearsWithCurrManager,DistanceFromHome))

cors <- cor(data_cont, use="pairwise")
corrplot(cors, type="upper", col=brewer.pal(n=8, name="RdYlBu"))

library(caret)

set.seed(999)
train_index <- createDataPartition(data$Attrition, p=0.70, list=FALSE)
train <- data[train_index,]
test <- data[-train_index,]

model_1 <- glm(Attrition~., data=data, family = binomial(link = 'logit'))
summary(model_1)
coef(model_1)
exp(coef(model_1))

# Making predictions and confusion matrix
# Train

probs_train <- predict(model_1, newdata=train, type='response')
#probs_train
predict <- as.factor(ifelse(probs_train >= 0.25,1,0))
#predict

confusionMatrix(predict, as.factor(train$Attrition))

# Test
probs_test <- predict(model_1, newdata=test, type='response')
#probs_test
predict_test <- as.factor(ifelse(probs_test >= 0.25,1,0))

confusionMatrix(predict_test,as.factor(test$Attrition))

str(predict_test)

library(pROC)
roc_1 <- roc(test$Attrition,probs_test)
plot(roc_1, col='purple', main="ROC")
auc(roc_1)



"""XG BOOST"""

head(data)

class(data)
# Convert categorical variables to factors
categorical_cols <- c("BusinessTravel", "Department", "EducationField", "Gender", "MaritalStatus", "OverTime")
data[categorical_cols] <- lapply(data[categorical_cols], as.factor)

# Perform one-hot encoding
data <- model.matrix(~.-1, data = data)

data <- as.data.frame(data)
head(data)

trainIndex <- sample(x = nrow(data), size = nrow(data) * 0.7)

train <- data[trainIndex,]
test <- data[-trainIndex,] # Returns all rows that are not in training set

head(train)
head(test)

train_m <- train[-c(2)]
test_m <- test[-c(2)]

train_y <- train$Attrition
test_y <- test$Attrition

library(xgboost)
model_xgboost <- xgboost(data = data.matrix(train_m), label = train_y, max.depth = 6, eta = 2, nthread = 4, nrounds = 2, objective = "binary:logistic")

y_pred_test <- predict(model_xgboost, data.matrix(test_m))

test$y_pred_prob <- y_pred_test

test$pred_test <- as.factor(ifelse(test$y_pred_prob >= 0.5,1,0))

head(test)

#table(data$Attrition)



confusionMatrix(test$pred_test,as.factor(test$Attrition))





# Final Project: Initial Analysis Report

# Exploratory Data Analysis

data <- read.csv("attrition.csv", header=TRUE,sep=",")

head(data)
summary(data)
str(data)
#describe(data)



#########################################

class(data)
# Convert categorical variables to factors
categorical_cols <- c("BusinessTravel", "Department", "EducationField", "Gender", "MaritalStatus", "OverTime")
data[categorical_cols] <- lapply(data[categorical_cols], as.factor)

# Perform one-hot encoding
data <- model.matrix(~.-1, data = data)

data <- as.data.frame(data)

head(data)

df <- data

###########################################
# Split data into train and test sets
###########################################
# seed() function in R is used to create reproducible results when writing code that involves creating variables 
# that take on random values. By using the set. seed() function, you guarantee that the same random values are produced 
# each time you run the code. # The value inside the seed() function does not matter 
set.seed(123)
trainIndex <- sample(x = nrow(df), size = nrow(df) * 0.7)

train <- df[trainIndex,]
test <- df[-trainIndex,] # Returns all rows that are not in training set

head(train)

train_x <- model.matrix(MonthlyIncome ~ ., train)[,-1]
test_x <- model.matrix(MonthlyIncome ~ ., test)[,-1]
# model.matrix is more advanced than just matrix function. 
# E.g. Model.matrix automatically converts categorical variables to dummy variables

head(train_x)
class(train_x)

colnames(train_x)

train_y <- train$MonthlyIncome
test_y <- test$MonthlyIncome

# Find best values of lambda
############################################################
# Find the best lambda using  corss-validation
set.seed(123)
#?cv.glmnet
cv.lasso <- cv.glmnet(train_x, train_y, alpha = 1, nfolds = 10) # alpha = 1 Lasso
plot(cv.lasso) # Values on top shows the number of non-zero coefficients
# Red dots are error estimates with their confidence interval
# dotted line on the left side represents the minimum value of Lambda; which retains 
#all 15 parameters
# dotted line on the right side represents maximum value within one standard error 
#of the minimum; it has 9 non-zero coefficients in the model. So it sets the coefficient of 
# one of the parameters to zero.

# lambda min - minimizes out of sample loss
# lambda 1se - largest value of lambda within 1 standard error of lambda min
log(cv.lasso$lambda.min)
log(cv.lasso$lambda.1se)

cv.lasso$lambda.min
cv.lasso$lambda.1se

cv.lasso$cvm # Returns Mean Square Error

plot(cv.lasso$cvm ~ cv.lasso$lambda)
plot(cv.lasso$cvm ~ log(cv.lasso$lambda))

############################################################
# Fit models based on lambda
############################################################
# Fit the final model on the training data using lambda.min
# alpha = 1 for Lasso (L2)
# alpha = 0 for Ridge (L1)

model.lasso <- glmnet(train_x, train_y, alpha = 1)
model.lasso

model.min <- glmnet(train_x, train_y, alpha = 1, lambda = cv.lasso$lambda.min)
model.min

# Display regression coefficients
coef(model.min)

# fit the final model on training data using lambda.1se
model.1se <- glmnet(train_x, train_y, alpha = 1, lambda = cv.lasso$lambda.1se)
model.1se

# Display regression coefficients
coef(model.1se)

# Make predictions on the test data using lambda.min
preds.train <- predict(model.1se, newx = train_x) # predict.glmnet
train.rmse <- rmse(train_y, preds.train) # 7.456

preds.test <- predict(model.1se, newx = test_x)
test.rmse <- rmse(test_y, preds.test)

# Compare rmse values
train.rmse 
test.rmse

# Make predictions on the test data using lambda.min
preds.train <- predict(model.min, newx = train_x) # predict.glmnet
train.rmse <- rmse(train_y, preds.train)
preds.test <- predict(model.min, newx = test_x)
test.rmse <- rmse(test_y, preds.test)

# Compare rmse values
train.rmse 
test.rmse

eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

eval_results(test_y, preds.test , test_x)

eval_results(train_y, preds.train , train_x)