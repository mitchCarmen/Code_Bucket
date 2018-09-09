# dplyr package
# 5 functions: select(), mutate(), filter(), arrange(), summarise()






###############################################################
####### 1) SELECT - selects certain VARIABLES to work with

select(dataFrame, c(Date, DepTime, ArrTime, TailNum))



###############################################################
####### 2) MUTATING - creating new variables

# REPLACEMENT

full$title[full$title == 'Mlle']        <- 'Miss'
full$title[full$title == 'Ms']          <- 'Miss'
full$title[full$title == 'Mme']         <- 'Mrs'
full$title[full$title == 'Capt']        <- 'Officer'
full$title[full$title == 'Col']        <- 'Officer'
full$ticket.size[full$ticket.unique == 1]   <- 'Single'
full$ticket.size[full$ticket.unique < 5 & full$ticket.unique>= 2]   <- 'Small'
full$ticket.size[full$ticket.unique >= 5]   <- 'Big'

# CREATING NEW

mutate(dataFrame , newVariable = (oldVariable1 + oldVariable2) /2 )
# Adding 2 variables at a time
mutate(dataFrame , newVariable = (oldVariable1 + oldVariable2) /2, newVar2 = var1 / var3 )

mutate(df, Date = paste(Year, Month, DayofMonth, sep = "-"))


###############################################################
####### 3) FILTER - selects certain OBSERVATIONS to work with

# Need to use the logical operators for filter generally
> , >= , < , <= , == , != , is.na , !is.na

filter(dataFrame, Variable1 >= 3000)
filter(hflights, (TaxiIn + TaxiOut) > AirTime
filter(dataFrame, Variable3 == c("Red","Blue","Green"))
filter(dataFrame, Variable3 %in% c("Red","Blue","Green")) # Same as above!

# "AND"
filter(hflights, DepDelay > 0 & Cancelled == 1)
filter(hflights, DepDelay > 0, Cancelled == 1) # Separated by comma is like an &

# "OR"
filter(hflights, DepTime < 500 | ArrTime > 2000)



###############################################################
####### 4) ARRANGE - Like Order
# Regarranges: factors by order level, characters by alphabetical

arrange(DF, Var1) # Arrange DF by Var1

arrange(DF, Var1, Var2) # Arragne DF by Var1, then by Var2

arrange(DF, Var1, desc(Var2)) # Descending order on the second Var


###############################################################
####### 5) SUMMARISE - Creating new data set to find results
# Uses any aggregator: min(), max(), mean(), median(), quantile(x,q), sd(), var(), IQR(), diff(range())

# Summarise by 2 variables
summarise(hflights, min_dist = min(Distance), max_dist = max(Distance))

# Combine with Filter
summarise(filter(dataFrame, Variable1 == 1), new_var = max(Variable10))




###############################################################
####### ONE HOT ENCODING

# Needed for XGboost
# This is the practice of splitting N levels into N columns represented by a 0 or 1
# This changes a matrix with categorical variables into a numerical/factors for MODELING
# For example, a RED, BLUE, GREEN variable will have to be split into 3 separate columns using 0/1s
# Use vtreat (from Datacamp) or dummies (from Yuri)
dummies::dummy.data.frame()
# Could also use: mMat <- model.matrix(formula, dataframe)
# vtreat() is the safest option when dealing with crazy data to be one-hot-encoded

# Using vtreat library
library(vtreat)
outcome <- "YVariable"
vars <- c("xVariable1","xVarialble2")
treatplan <- designTreatmentsZ(OrigDF, vars)
scoreFrame <- treatplan %>%
    magrittr::use_series(scoreFrame) %>%
    select(varName, origName, code)
newvars <- scoreFrame %>%
    filter(code %in% c("clean","lev")) %>%
    use_series(varName)
dframe.treated <- prepare(treatplan, OrigDF, varRestriction = newvars)
# Do the dframe.treated for Train and Test
