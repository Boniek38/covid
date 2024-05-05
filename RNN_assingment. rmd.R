```{r}
#libraries to be used
library(datasets)
library(nnet)
library(caret)
library(dplyr)
library(xts)
library(lubridate)
library(lubridate)
library(Metrics)
library(ggplot2)

#Loading data in R
data <- read.csv("C:/postgraduate/sem 3/non parametric regression analysis/time_series_covid19_confirmed_global_narrow.csv", header=TRUE, comment.char="#")
head(data)
data[5]


#filtering Kenya Data and removing the zeros
df = select(data, c(2,5:6)) %>% filter(data$Country.Region == "Kenya")
head(df)
df_new <- df[apply(df != 0, 1, all),] 


#taking cummulative values and Dates from kenyan data
value <- df_new$Value
Dates <- Dates<-as.Date(df_new$Date, format = "%m/%d/%Y")
class(Dates)
Cum_cases <- data.frame(Dates, value)
class(Cum_cases$Dates)

#changing the data frame to a time series data
Cum_cases <- as.xts(Cum_cases)
plot(Cum_cases)
head(Cum_cases)



#Normalizing the data using standard normal since the data is so spread
normalize_zscore <- function(x, mea_n, sdx) {
  (x - mea_n) / sdx
}
mu <- mean(Cum_cases)
stdx <- sd(Cum_cases)
Cum_cases <- normalize_zscore(Cum_cases, mu, stdx)

#Finding x_t_1, x_t_2, x_t_3 using the lag function
x_t_1<-lag.xts(Cum_cases,1)
x_t_2<-lag.xts(Cum_cases,2)
x_t_3<-lag.xts(Cum_cases,3)
y<-Cum_cases

#combining the data using cbind function
lag_data<-cbind(y,x_t_1,x_t_2,x_t_3)
head(lag_data)

#Traing the model
Training_Sample <- lag_data[4:(0.75*nrow(lag_data)),]
sample_predictor_training <- lag_data[4:(0.75*nrow(lag_data)),2:4]
summary(Training_Sample)

#Fitting the nnet model with 6 hidden nodes iin 1 hidden layer
nnet.model <- nnet(value ~ value.1 + value.2 + value.3, data = Training_Sample, 
                   size = 25, rang = 0.1, decay = 5e-04, maxit = 1000, trace = FALSE , linout = TRUE)

#in the sample statistices
standardized_fitted_values <-  nnet.model$fitted.values
predicted_values_for_training <-  predict(nnet.model, sample_predictor_training, type = "raw")
denormalize <- function(x, mea_n, sdx){
  (x*sdx) + mea_n
}
predicted_values_for_training <- denormalize(predicted_values_for_training, mu, stdx)
True_values_for_training <- lag_data[4:(0.75*nrow(lag_data)),1]
True_values_for_training <- denormalize(True_values_for_training, mu, stdx)
(head(data.frame(round(predicted_values_for_training,0),True_values_for_training)))

#Function for R-square 
rsquare <- function(y_hat, y){
  y_bar <-  mean(y)
  numerator <- (sum((y-y_bar)*(y_hat-y_bar)))^2
  denominator <- (sum((y-y_bar)^2))*(sum((y_hat-y_bar)^2))
  return (numerator/denominator)
}

#Calculating R-squared
R_Squared_for_training <- rsquare(predicted_values_for_training, True_values_for_training)
R_Squared_for_training

#outside the sample statistics
Testing_Sample<- lag_data[(0.75*nrow(lag_data)+1):nrow(lag_data),]
predicted_values_for_Testing <-  predict(nnet.model, newdata = Testing_Sample, type = "raw")
denormalize <- function(x, mea_n, sdx){
  (x*sdx) + mea_n
}
predicted_values_for_Testing <-denormalize(predicted_values_for_Testing,mu,stdx)
True_values_for_Testing <- lag_data[(0.75*nrow(lag_data)+1):nrow(lag_data),1]
denormalize <- function(x, mea_n, sdx){
  (x*sdx) + mea_n
}
True_values_for_Testing <-denormalize(True_values_for_Testing,mu,stdx)
head(data.frame(predicted_values_for_Testing,True_values_for_Testing))


#Calculating R-squared
R_Squared_for_testing <- rsquare(predicted_values_for_Testing, True_values_for_Testing)
R_Squared_for_testing

#finding the fitted values
fitted<-nnet.model$fitted.values
denormalize <- function(x, mea_n, sdx){
  (x*sdx) + mea_n
}

predicted_values_for_Testing <- as.data.frame(predicted_values_for_Testing)
predicted_values_for_Testing$Dates <- rownames(predicted_values_for_Testing)
predicted_values_for_Testing$Dates <- ymd(predicted_values_for_Testing$Dates)

True_values_for_Testing <- as.data.frame(True_values_for_Testing)
True_values_for_Testing$Dates <- rownames(True_values_for_Testing)
True_values_for_Testing$Dates <- ymd(True_values_for_Testing$Dates)

#plotting true values used for testing and predicted values for testing
plot(True_values_for_Testing$Dates, True_values_for_Testing$value, type = "l", col = 1)
lines(predicted_values_for_Testing$Dates, predicted_values_for_Testing$V1, type = "l", col = 2)
legend("bottomright", c("true values used for testing", "predicted values for testing"), lty = 1, col=1:2)

#extracting dates for the predicted training data
predicted_values_for_training <- as.data.frame(predicted_values_for_training)
predicted_values_for_training$Dates <- rownames(predicted_values_for_training)
predicted_values_for_training$Dates <- ymd(predicted_values_for_training$Dates)


#combining the predicted training data and predicted testing data to one dataset
combined_predicted_data <- predicted_values_for_training %>% rbind(predicted_values_for_Testing)
class(as.Date(combined_predicted_data$Dates))

plotting_data <- NULL
plotting_data$dates <- mdy(df$Date) 
plotting_data$values <- df$Value
plotting_data <- as.data.frame(plotting_data)
plotting_data <- plotting_data[order(as.Date(plotting_data$dates, format ="%m/%d/%Y")),]

#plotting both true Data and  the Predicted Data in the same graph

plot(plotting_data$dates, plotting_data$values, type = "l", col = 1,
     ylab = "Covid_Cases Cumulative", xlab = "Dates")
title(main = "Covid_Cases Cumulative by Their Date")
lines(as.Date(combined_predicted_data$Dates), combined_predicted_data$V1, type = "l", col = 2)
legend("bottomright", c("True Data", "Predicted Data"), lty = 1, col=1:2)

#M(X)_hat after denormalizing
denormalize(nnet.model$fitted.values,mu,stdx)


#plottting the model
library(NeuralNetTools)
plotnet(nnet.model)
summary(nnet.model)
```

