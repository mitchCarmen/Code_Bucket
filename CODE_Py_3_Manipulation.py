###############################################################
############### RESOURCES
https://developers.google.com/edu/python/introduction
https://docs.python.org/3/tutorial/index.html


###############################################################
####################### GETTING STARTED

# SLICING
# BOOLEAN INDEXING
# BOOLEAN REPLACEMENT
# CREATING NEW COLUMN
# RENAMING COLUMNS
# APPLY
# DUPLICATE DATA
# MISSING DATA
# REGEX
# CROSSTAB
# CONCATENATING DFs
# MERGING DATAFRAMES
# SORTING DATAFRAMES
# MELTING DATAFRAMES
# PIVOTTING DATA
# PIVOT TABLE
# CUT FUNCTION FOR BINNING
# ONE HOT ENCODING
# SCALING DATA

###################### SLICING

# SLICING
# Get the ith row
data[i]
# Get the column named 'Fare'
data['Fare']
# Get the first 3 entries
data[:3]



word = 'Python'

word[0]
'p'

word[-1]
'n'

word[1:4]
'yth'

word[ :2]
'Py'

word[-2: ]
'on'

word[ :2] + word[2:]
'Python'


# SLICING COLUMNS
# df.iloc uses indeces numbers
df.iloc[row_indices, column_indices]
# df.loc uses labels
df.loc[row_labels, column_labels]

df = pd.DataFrame({
  'A': [1,2,3],
  'B': [4,5,6],
  'C': [7,8,9]},
  index = ['x','y','z'])

df['A']
# This is the same thing...
df.A
# This is the same thing... if we are using indeces
df.iloc[0]
df[['A','B']] # Pass list of column names inside square brackets for multiple column subsetting



# SLICING ROWS
# df.iloc uses indeces numbers
# df.loc uses labels

df.loc['x'] # Selects the first row
df.loc['men','age'] # Selects 'men' row by the 'age column'
df.iloc[0] # Selects the first row
df.iloc[[0,1]] # Selects multiple rows and all columns
df.iloc[0, :] # Selects first row and all columns
df.iloc[[0,1], :] # Selects first two rows and all columns

df.loc['Jan':'Apr', : ] # Slices rows for months Jan-April, and ALL Columns
df.loc[:, 'product1':'product3'] # Slices ALL ROWS, but products 1-3


# SUBSETTING ROWS AND COLUMNS
df.loc[['x','y'],['A','B']] # Selects Rows X & Y and Columns A & B; Lists must be passed
# Subset rows and columns
print(tips.loc[tips['sex'] == 'Female', ['total_bill', 'tip', 'sex']])
# 3 rows and 3 columns with iloc
tips.iloc[:3, :3]


# COULD ALSO
df['salt']['Jan']
# This will first index through the Sale COLUMN, then through the JAN ROW.

###################### BOOLEAN INDEXING

# List all females who did not graduate and got a loan
data.loc[(data["Gender"]=="Female") & (data["Education"]=="Not Graduate") & (data["Loan_Status"]=="Y"), \
  ["Gender","Education","Loan_Status"]]

df[df.A == 3] # Will return a df where Variable A has 3 in it

# Print all the rows where sex is Female
print(tips.loc[tips.sex == 'Female'])

df[(df.A == 3) | (df.B == 4)] # Selects where Var A is 3 OR where Var B is 4. '&' is for AND


###################### BOOLEAN REPLACEMENT

df[df == '?'] = np.nan

df[df.var1 == 'Doctor'] = 'DR'


###################### CREATING NEW COLUMN

# simply make it!
# 'sex' did not exist but then we...
df['sex'] = df.variable1.str[0] # looks at the first element of the variable column to make 'm' or 'f'


###################### CREATING NEW DF

# List within the index [ ]  is how we create sub sequent DFs
# Assume DF has 10 variables
new_df = DF[['var1','var2','var3','var4']]


###################### RENAMING COLUMNS

df.columns = ['name1','name2','name3']


###################### APPLY

# Create a new function:
def num_missing(x):
  return sum(x.isnull())

#Applying per column:
print "Missing values per column:"
print data.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column
#Applying per row:
print "\nMissing values per row:"
print data.apply(num_missing, axis=1).head() #axis=1 defines that function is to be applied on each row

# Column applications
df.apply(np.mean, axis=1) 
# Row Applications
df.apply(np.mean, axis=0) 

# Write the lambda function using replace to remove the '$'
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))
# EXACT SAME BELOW...
# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])


###################### DUPLICATE DATA

df = df.drop_duplicates() # Drops exact dupe of rows

###################### MISSING DATA
# start with df.info() to see variables, counts, and possible missing

print(df.isnull().sum())
# count the null by column

# CONVERT MISSING to nan
df.variable1.replace(0, np.nan, inplace = True) # Replaces 0 value with nan

# DROP ROWS OF MISSING
df = df.dropna() # This removes full row if a single element is missing tho. Not great.

# FILLna with 0: We can fill nan this way
df[['variable1','variable2']] = df[['variable1','variable2']].fillna(0)

# FILLna with Statistic
mean_value = tips['tip'].mean()
tips['tip'] = tips['tip'].fillna(mean_value)

## FINAL STEP: Assert Data



# IMPUTE MISSING
#First we import a function to determine the mode
from scipy.stats import mode
mode(data['Gender'])
mode(data['Gender']).mode[0]
#Impute the values:
data['Gender'].fillna(mode(data['Gender']).mode[0], inplace=True)
data['Married'].fillna(mode(data['Married']).mode[0], inplace=True)
data['Self_Employed'].fillna(mode(data['Self_Employed']).mode[0], inplace=True)
#Now check the #missing values again to confirm:
print data.apply(num_missing, axis=0)

# IMPUTER
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #1 means rows, 0 means columns. Could also use 'most_frequent' as strategy
imp.fit(X)
X=imp.transform(X)
# This might not be the most efficient method... Instead use the below PIPELINE method!!

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #1 means rows, 0 means columns
logreg = LogisticRegression()
steps = [('imputation', imp),
          ('logistic_regression', logreg)]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(type in the args of this function!)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
pipeline.score(X_test, y_test)


###################### REGEX -- See regex file for more
# STRING MANIPULATION Library = 're'

import re
string = the recipe calls for 10 strawberries and 1 banana
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')
print(matches)
['10', '1']

pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1)

pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))
print(pattern2)

pattern3 = bool(re.match(pattern='[A-Z]\w*', string='Australia'))
print(pattern3)


import re
#%% clean data
def clean_text(mystring): 
    mystring = re.sub(r"@\w+", "", mystring) #remove twitter handle
    mystring = re.sub(r"\d", "", mystring) # remove numbers  
    mystring = re.sub(r"_+", "", mystring) # remove consecutive underscores
    mystring = mystring.lower() # tranform to lower case    
    
    return mystring.strip()
mydata["tweet_text_cleaned"] = mydata.tweet_text.apply(clean_text)


###################### CROSS TAB

# NUMBERS
pd.crosstab(data["Credit_History"],data["Loan_Status"],margins=True)

# PERCENTS
def percConvert(ser):
    return ser/float(ser[-1])
    pd.crosstab(data["Credit_History"],data["Loan_Status"],margins=True).apply(percConvert, axis=1)


###################### CONCATENATING DFs
# Combinging 2 DFs into a single DF
# Using Pandas (axis=0 is default for row-wise concatenation/ use axis=1 for column wise)
concatenated = pd.concat([df1, df2], ignore_index = True) # on top of one another with reset index
concatenated = pd.concat([df1, df2], axis=1) # for column wise concatenation

# Concatenating Many Files is "Globbing"
import glob
csv_files = glob.glob('*.csv') # Searches with * wildcard to find any name that is a .csv file. Conversely, '?' is a wildcard for any single char
list_data = []
for filename in csv_files:
  data = pd.read_csv(filename)
  list_data.append(data)
pd.concat(list_data)


###################### MERGING
# 1-1; Many-1 (1 to Many); Many to Many
# Merge
data_merged = data.merge(left=newDF,
  right=oldDF,
  on=None,
  left_on='Var1',
  right_on='Var2thatMatchesVar1',
  sort=False)

# 1-1
merged = pd.merge(left=site, right=visited, left_on='name', right_on='site')

# M-M: Each df does not have unique keys for a merge-- every pairwise combo is created



###################### SORTING DF

data_sorted = data.sort_values(['ApplicantIncome','CoapplicantIncome'], ascending=False)
data_sorted[['ApplicantIncome','CoapplicantIncome']].head(10)


###################### MELTING DATAFRAMES (Inverse of Pivoting)
# Turn columns into rows
# Melting an untidy DF to a tidy DF by collapsing two separate treatment columns to 1 column
pd.melt(frame=df, id_vars='name',           # Use brackets for multiple columns
        value_vars=['treatment a', 'treatment b'],
        var_name='treatment', value_name='result')
# id_vars are the columns we do NOT want to melt
# value_vars are the columns we DO want to melt
# var_name and value_name are the new columns

###################### PIVOTING (Inverse of MELTING) *Can't deal with duplicate values tho so pivot_table instead
# Turn unique values into separate columns
weather_tidy = weather.pivot(index='date',
                              columns='element',
                              values='value')


###################### PIVOT TABLE (Use when Pivoting does NOT work)

weather_tidy = weather.pivot_table(index='date',
                              columns='element',
                              values='value',
                              aggfunc=np.mean) # Default aggfunc is np.mean by the way!
# Index represents the column to not change
# Columns the columns we want to pivot by
# Values stipulates the values to fill our columns once we pivot
# AggFunc the function we will use to aggregate IF there are multiple values

#Determine pivot table
impute_grps = data.pivot_table(values=["LoanAmount"], index=["Gender","Married","Self_Employed"], aggfunc=np.mean)
print impute_grps





############################################################
######################## pre processing 
############################################################

###################### BINNING

#Binning:
def binning(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()
  #create list by adding min and max to cut_points
  break_points = [minval] + cut_points + [maxval]
  #if no labels provided, use default labels 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)
  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin

#Binning age:
cut_points = [90,140,190]
labels = ["low","medium","high","very high"]
data["LoanAmount_Bin"] = binning(data["LoanAmount"], cut_points, labels)
print pd.value_counts(data["LoanAmount_Bin"], sort=False)
sadffrom keras.utils import to_categorical
data = pd.read_csv('basketball_data.csv')
predictors = data.drop(['shot_result'], axis = 1).as_matrix()
target = to_categorical(data.shot_results)


###################### ONE HOT ENCODING
# Can use scikit learn: OneHotEncoder() or get_dummies in Pandas
import pandas as pd
df = pd.read_csv('autoData.csv')
df_dummy = pd.get_dummies(df) 


###################### SCALING DATA

# Standardized: subtract the mean and divide by variance
#       - features are centered around zero and have var 1
#       - Go from -1 to 1

# Normalized: Range from 0 to 1

from sklearn.preprocessing import scale
X_scaled = scale(X)
# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))
# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.std(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))











