###############################################################
############### RESOURCES
https://developers.google.com/edu/python/introduction
https://docs.python.org/3/tutorial/index.html

###############################################################

# Getting Started
# Statistics
# Visualizations





###############################################################
####################### GETTING STARTED

df = pd.read_csv('data.csv', header=None)
df.head()
df.tail() 

df.columns
df.index

df.shape # Use without parenthesis
df.info() # Frequency Counts

# Get COUNTS of variables
df.variable1.value_counts(dropna=False) # instead of bracket notation
df['variable1name'].value_counts(dropna=False).head() # showing only the head


###############################################################
####################### STATISTICS

# SUMMARY STATISTICS
df['fare'].describe()
df.fare.describe()

# Find MIN
print(df['Engineering'].min())
# Find MAX
print(df['Engineering'].max())
# Construct the mean percentage per year: mean
mean = df.mean(axis='columns')
# Plot the average percentage per year
mean.plot()
# Display the plot
plt.show()


iris.median() # returns median
df['mpg'].median  # another example


iris.mean()

q = .5  # quantile searching for the 50% mark or median
iris.quantile([q])
## OR for the 25% and 75% quantiles
iris.quantile([0.25, 0.75])





###############################################################
###################### VISUALIZATIONS

# BAR PLOTs for discrete data
# HISTOGRAMs for continuous data


# HISTOGRAM - of one column of data frame
pd.DataFram.hist(df[['col1name']])
plt.xlabel('Extent of Stuff')
plt.ylabel('Number of Countries')
plt.show()

# HISTOGRAM
import matplotlib.pyplot as plt
df.poplulation.plot('hist')
plt.show()


y_columns = (['AAPL','IBM'])
df.plot(x='Month', y=y_columns)
plt.title('Monthly Stock Prices')
plt.ylabel('Price ($USD)')
plt.show()


## SCATTER PLOT: For relationships between columns
df.plot(kind='scatter', x='hp', y='mpg', s=sizes)
# Add the title
plt.title('Fuel efficiency vs Horse-power')
# Add the x-axis label
plt.xlabel('Horse-power')
# Add the y-axis label
plt.ylabel('Fuel efficiency (mpg)')
# Display the plot
plt.show()


# SEPARATE BOX PLOTS
cols = (['weight', 'mpg'])
# Generate the box plots
df[cols].plot(kind='box', subplots=True)
# Display the plot
plt.show()

# BOX PLOT: Separate box plot by each continent
df.boxplot(column='population', by='continent')
plt.show()

# SEPARATE STACKED BOX PLOTS
# Display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows=3, ncols=1)
# Generate a box plot of the fare prices for the First passenger class
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')
# Generate a box plot of the fare prices for the Second passenger class
titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y='fare', kind='box')
# Generate a box plot of the fare prices for the Third passenger class
titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y='fare', kind='box')
# Display the plot
plt.show()


# Separate PDF and CDF plots
# This formats the plots such that they appear on separate rows
fig, axes = plt.subplots(nrows=2, ncols=1)
# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0,.3))
plt.show()
# Plot the CDF
df.fraction.plot(ax=axes[1], kind='hist', bins=30, normed=True, cumulative=True, range=(0,.3))
plt.show()




