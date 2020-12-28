

# To Avoid Large Losses:
# 1) Seek diversified portfolios


#1) Portfolio Weights & Returns
#2) Portfolio Performance Evaluation
#3) Drivers of Performance
#4) Portfolio Optimization

# Important Packages: TS packages(xtz, zoo), PerformanceAnalytics

### WEIGHTS ###
###############
# Define marketcaps
 marketcaps = c(5,8,9,20,25,100,100,500,700,2000)

# Compute the weights
weights = marketcaps/sum(marketcaps)

# Inspect summary statistics
summary(weights)

# Create a barplot of weights
barplot(weights)

### RETURNS ###
###############
# Vector of initial value of the assets
in_values <- c(1000,5000,2000)

# Vector of final values of the assets
fin_values <- c(1100,4500,3000)

# Weights as the proportion of total value invested in each assets
weights <- in_values/sum(in_values)

# Vector of simple returns of the assets
returns <- (fin_values - in_values)/in_values

# Compute portfolio return using the portfolio return formula
preturns <- sum(weights*returns)

 ### EVALUATION ###
 ##################
# Return.calculate(xts(prices)) : To compute the asset returns
# TS of prices as XTS object in Date Structure YYYY-MM-DD
# You get a TS of returns as a result of feeding prices

# Return.portfolio() : To compute the portfolio return
# Need to other set the initial weights & NOT intervene OR do rebalancing
# Takes R, weights, and rebalance_on arguments


# NEW CREATION
# Create the weights
eq_weights <- c(.5, .5)

# Create a portfolio using buy and hold
pf_bh <- Return.portfolio(R = returns, weights = eq_weights)

# Create a portfolio rebalancing monthly
pf_rebal <- Return.portfolio(R = returns, weights = eq_weights, rebalance_on = "months")

# Plot the time-series
par(mfrow = c(2, 1), mar = c(2, 4, 2, 2))
plot.zoo(pf_bh)
plot.zoo(pf_rebal)


# NEW CREATION
# Create the weights
eq_weights <- c(.5,.5)

# Create a portfolio using buy and hold
pf_bh <- Return.portfolio(returns, weights = eq_weights, verbose = TRUE )

# Create a portfolio that rebalances monthly
pf_rebal <- Return.portfolio(returns, weights = eq_weights, rebalance_on = "months", verbose = TRUE )

# Create eop_weight_bh
eop_weight_bh = pf_bh$EOP.Weight

# Create eop_weight_rebal
eop_weight_rebal = pf_rebal$EOP.Weight

# Plot end of period weights
par(mfrow = c(2, 1), mar=c(2, 4, 2, 2))
plot.zoo(eop_weight_bh$AAPL)
plot.zoo(eop_weight_rebal$AAPL)


# NEW CREATION
# Convert the daily frequency of sp500 to monthly frequency: sp500_monthly
sp500_monthly <- to.monthly(sp500)

# Print the first six rows of sp500_monthly
head(sp500_monthly)

# Create sp500_returns using Return.calculate using the closing prices
sp500_returns <- Return.calculate(sp500_monthly[,4])

# Time series plot
plot.zoo(sp500_returns)

# Produce the year x month table
table.CalendarReturns(sp500_returns)


# NEW CREATION
# Compute the mean monthly returns
mean(sp500_returns)

# Compute the geometric mean of monthly returns
mean.geometric(sp500_returns)

# Compute the standard deviation
sd(sp500_returns)


# NEW CREATION--SHARPE RATIO
# Compute the annualized risk free rate
annualized_rf <- (1 + rf)^12 - 1

# Plot the annualized risk free rate
plot.zoo(annualized_rf)

# Compute the series of excess portfolio returns
sp500_excess <- sp500_returns - rf

# Compare the mean
mean(sp500_returns)
mean(sp500_excess)

# Compute the Sharpe ratio
sp500_sharpe <- mean(sp500_excess) / sd(sp500_returns)

# Compute the annualized mean
Return.annualized(sp500_returns)

# Compute the annualized standard deviation
StdDev.annualized(sp500_returns)

# Compute the annualized Sharpe ratio: ann_sharpe
ann_sharpe <- SharpeRatio.annualized(sp500_returns)

# Compute all of the above at once using table.AnnualizedReturns()
table.AnnualizedReturns(sp500_returns)


# NEW CREATION
# Calculate the mean, volatility, and sharpe ratio of sp500_returns
returns_ann <- Return.annualized(sp500_returns)
sd_ann <- StdDev.annualized(sp500_returns)
sharpe_ann <- SharpeRatio.annualized(sp500_returns, Rf = rf)

# Plotting the 12-month rolling annualized mean
chart.RollingPerformance(R = sp500_returns, width = 12, FUN = "Return.annualized")
abline(h = returns_ann)

# Plotting the 12-month rolling annualized standard deviation
chart.RollingPerformance(R = sp500_returns, width = 12,  FUN = "StdDev.annualized")
abline(h = sd_ann)

# Plotting the 12-month rolling annualized Sharpe ratio
chart.RollingPerformance(R = sp500_returns, width = 12, FUN = "SharpeRatio.annualized", Rf = rf)
abline(h = sharpe_ann)


# TO CHART ALL THREE IN ONE GRAPH
charts.RollingPerformance(R = sp500_returns, width = 12)


# NEW CREATION
