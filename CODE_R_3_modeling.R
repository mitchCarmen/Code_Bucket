###############################################################
    ####### TRAINING DATA METHODS #################################

#### TRAINING AND HOLDOUT VALIDATION
N <- nrow(data)
target <- round(.75 * N)
set.seed(1)
gp <- runif(N)
train <- data[gp < .75, ]
test <- data[gp >= .75, ]


#### CROSS VALIDATION
# 1) REGRESSION EXAMPLE
vtreat::kWayCrossValidation()
set.seed(1)
splitPlan <- kWayCrossValidation(nRows, nSplits, dframe, y)

k <- 3 # Number of folds
mpg$pred.cv <- 0 # Initialize a column of 0 length
for(i in 1:k) {
  split <- splitPlan[[i]]
  model <- lm(cty ~ hwy, data = mpg[split$train,])
  mpg$pred.cv[split$app] <- predict(model, newdata = mpg[split$app, ])
}
# Predict from a full model
mpg$pred <- predict(lm(cty ~ hwy, data = mpg))
# Get the rmse of the full model's predictions
rmse(mpg$pred, mpg$cty)
# Get the rmse of the cross-validation predictions
rmse(mpg$pred.cv, mpg$cty)
# THIS CV SIMPLY PREDICTS HOW WELL A MODEL BUILT FROM ALL THE DATA WILL PERFORM ON NEW DATA.
# THE CV AND FULL NUMBERS SHOULD BE CLOSE.

# 2) CATEGORICAL EXAMPLE


#### ENSEMBLE METHODS
# 1) Random Forest

# 2) XGboost (Boosted Trees)







###############################################################
####### MODELING METHODS ######################################

#### LINEAR REGRESSION - minimizes squared errors
FIT <- lm(y ~ x + x2, data = dataFrame)
# Beta coefficients (+/-) will show how Y moves from X

#### LOGISTIC REGRESSION
FIT <- glm(y ~ x + x2, data = data, family = binomial)
# Not the same BETA INTERPRETATION as linear

#### POISSON REGRESSION / QUASIPOISSON
FIT <- glm(y ~ x + x2, data = data, family = quasipoisson)

### GAM - Genearlized Additive Method - Learns NON-Linear functions easily!
# GREAT when you dont know how to transform the data or dont have domain knowledge!!
# Can use gaussian / binomial / poisson as the family
# GAMs are more likely to overfit and therefore better on large datasets
FIT <- gam(y ~ s(x) + x2, data = dframe, family = above listed type)
# Put s() on any CONTINUOUS variable that you are not sure about it needing a transformation
# Leaving any other variable without and s() will assume a linear relationship

### DECISION TREE
# Good for regression or classification
# Good for linear or nonLinear


### RANDOM FOREST (Ensemble Method)
# Reduces overfit / Increases model expressiveness / Finer grain predictions
# Average result is computed from all the trees grown
FIT <- ranger::ranger(formula, dataframe, num.trees = 500, respect.unordered.factors = "order", seed = set.seed())


### GRADIENT BOOSTED TREES (Ensemble Method)
# Cant use categorical data OR dataframes -- make it all numerics and a matrix
cv.FIT <- xgboost::xgb.cv(data = as.matrix(originalDF_treated_by_HOT_encoding),
                        label = originalDF$Y,
                        nrounds = 500,
                        nfold = 3,
                        objective = "reg:linear",
                        eta = 0.01,
                        max_depth = 10,
                        early_stopping_rounds = 10,
                        verbose = F)
elog <- cv.FIT$evaluation_log
ntrees <- which.min(elog$test_MSE)
FIT <- xgboost(data = as.matrix(originalDF_treated_by_HOT_encoding),
                   label = originalDF$Y,  # column of outcomes
                   nrounds = ntrees,       # number of trees to build
                   objective = "reg:linear",
                   eta = .01,
                   depth = 10,
                   verbose = 0  # silent)

###############################################################
####### PREDICTION ############################################

# Linear Regression
predict(model)
predict(model, newdata)

# Binomial / Poisson
predict(model, newdata, type = "response")

# GAM
predict(model, newdata, type = "terms") # To get the Y values on the model
predict(model, newdata, type = "response") # To predict traditionally
# Depending on the situation, GAM predicts() into a matrix so sometimes it is needed to as.numeic(predict())
as.numeric(predict(model, newdata, type = "response")) # To predict into vector of numerics

# RANDOM FOREST (Ensemble Method)
# Make predictions on the August data
originaDF$pred <- predict(FIT_rf, originalDF)$predictions # Using ranger::ranger() RF fit

# GRADIENT BOOSTED TREES (Ensemble Method)
originaDF$preds <- predict(FIT.xgb, as.matrix(originalDF_treated_by_HOT_encoding))


###############################################################
####### EVALUATION#############################################
# package ModelMetrics is super helpful!
# Has all regular metrics to evaluate a model

# Can use:
evaluate_model() # put the model fit into the function

# Can also hold certain variables constant to look at impacts:
evaluate_model(model)
evaluate_model(model, age = 21) # Will show the effect of holding age constant


####### EVALUATION PROCESS

# 1) Look at model details
summary(model)
# Try broom::glance() instead of summary()

####### For REGRESSION

# 2) Look at prediction residuals compared to actual
ggplot(unemployment, aes(x = predictions, y = residuals)) +
    geom_point()

# 3) Can also look at a GainPlot
WBPlots::GainCurvePlot(dataframe, "predictions", "actual", "Title")
# Shows how well the model is sorting high / low observations
# Real nice for LOGISTIC REGRESSION / QUASIPOISSON / POISSON

# 4) RMSE - Root Mean Squared Error
# Get the residuals, square them, take the mean, find the sqrt()
# "Prediction Error"
# Compare to the STDEV of the actual Y
res <- unemployment$female_unemployment - unemployment$predictions
rmse <- sqrt(mean(res^2))
# DO THIS TO CALCULATE AND VISUALIZE
# Calculate the RMSE of the predictions
originalDF %>%
  mutate(residual = actual - pred)  %>% # calculate the residual
  summarize(rmse  = sqrt(mean(residual^2)))      # calculate rmse
# Plot actual outcome vs predictions (predictions on x-axis)
ggplot(originalDF, aes(x = pred, y = actual)) +
  geom_point() +
  geom_abline()


# 5) R2 & Adj R2 - Values between 0 and 1
# R2 = 1 - (RSS / SST)
# RSS is variance from model / SST is total sum of squares from data
(fe_mean <- mean(unemployment$female_unemployment))
(tss <- sum((unemployment$female_unemployment - fe_mean)^2))
(rss <- sum(unemployment$residuals^2))
(rsq <- 1 - (rss/tss))
    # Additionally (for interviews), The R2 is also the cor between the predictions and the Y.actual for models that minimize the squared error
    # (cor(predictions, actual))^2
# Additionally, here is the function:


####### For LOGISTIC REGRESSION / POISSON

# 3) Can also look at a GainPlot
WBPlots::GainCurvePlot(dataframe, "predictions", "actual", "Title")
# Shows how well the model is sorting high / low observations
# Real nice for Logistic REGRESSION

# 5) # Calculate pseudo-R-squared # Just like R2 for the logit model but instead of variance explained it is "deviance explained"
# Use broom::glance() for the summary of the model
pseudoR2 <- 1 - (model$deviance/model$null.deviance))
