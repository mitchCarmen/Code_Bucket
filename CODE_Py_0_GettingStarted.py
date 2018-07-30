###############################################################
############### RESOURCES
https://developers.google.com/edu/python/introduction
https://docs.python.org/3/tutorial/index.html


# GETTING STARTED
# Numpy wants arrays of all the same type generally
# Pandas is for the DataFrame solution

# 0 : BASIC TYPES
#       1. tuple
#       2. lists
#       3. sets
#       4. dictionaries
#       5. CONVERTING TYPES
# 1 : READING DATA IN ALL FORMATS
# 2 : GET REQUESTS: READING DATA FROM THE WEB


#########################################
######################################### BASIC TYPES
#########################################


# TUPLE : Immutable meaning not modifiable / easier to procees cuz less memory usage
even_nums = (2,4,6) # () initiates a tuple 
# Tuple Zipping
pairs = zip(girl_names, boy_names) # Zips two variables into single variable tuple
# Tuple Unpacking
for idx, pair in enumerate(pairs):
    # Unpack pair: girl_name, boy_name
    girl_name, boy_name = pair
    # Print the rank and names associated with each rank
    print('Rank {}: {} and {}'.format(idx, girl_name, boy_name))



# LISTS : [] initiates a list are are MUTABLE
squares = [1, 4, 9, 16]
sqaures + [25, 36]
df.extend() # adds more to the list
df.index('Mitch') # finds the position of 'Mitch' in the list of names
df.pop(index()) # to remove item from list


# SETS : Used for storing UNIQUE data in an unordered manner. MUTABLE
        # Created from a list generally
cookie_types = set()
cookie_types = ['choco','vanialla','oatmeal']
cookie_types.add('raison') # will only add if it's unique
cookie_types.update(['choco','caramel']) # merges in another set or list
cookie_types.discard() # removes an element from a set
cookie_types.pop() # removes element from set
cookie_types.union(b) # set method to return all names (or). A union B.
cookie_types.intersection() # set method to return overlapping data (and). A intersect B.
cookie_types.difference() # set method to find differences between sets


# DICTIONARIES : Best storing key/value pairs. Can be nested. Iterable.
art = {}
for name, zipcode in galleries: # galleries was a tuple and we want to transform to a dictionary for ease
    art[name] = zipcode
for name in art:
    print(name)

dictionary.keys() # shows all KEYS
dictionary.get() # searches for KEYS in the dict or says "NOT FOUND"
print(dictionary['KEY4'])

# add data to dict by...
galleries = {}
gal_list_123 = [('KeyA', '12345'),('KeyB', '45678')]
galleries['KEY123'].update(gal_list_123)

# remove key/value from dict by...
del & .pop() # Safer than Del

del art_galleries['1234'] # will throw error if doesnt exist
galleries_1031 = art_galleries.pop['1031']


##################
################## CONVERTING TYPES
##################

print(df.dtypes)
type()
# type of each column
df.info()

# Converts A to string 
df['A'] = df['A'].astype(str)
df['gender_cat'] = df['gender'].astype('category')
df['treament_a'] = pd.to_numeric(df['treament_a'], errors = 'coerce') # Converting '-' into numeric type



#########################################
######################################### READING FROM SOURCES
#########################################

# READING PLAIN TEXT FILE
filename = 'mitchcarmen.txt'
file = open(filename, mode = 'r') # r is for read, use 'w' for writing
text = file.read()
file.close() # best practice to close this connection
print(text)


# READING FLAT FILE
# Use numpy(for numerical arrays) or pandas
import numpy as np
filename = 'MNIST.txt'
data = np.loadtxt(filename, delimiter = ',')
data
# to skip first row due to header
data = np.loadtxt(filename, delimiter = ',', skiprows = 1, usecols=[0,2]) #use cols only grabs 1st and 3rd here
# print the 10th element
print(data[9])


# READING MULTI DATATYPE FILES
data = np.genfromtxt('titanic.csv', delimiter = ',', names = True, dtype = None)
# names says there is a header
# Numpy wants arrays of all the same type generally
# Pandas is for the DataFrame solution


# Import a CSV using Pandas
import pandas as pd
filename = 'qualitymeat.csv'
data = pd.read_csv(filename)
data = pd.read_csv(filename, sep = ',', nrows = 5, header = None) # Just another example
data.head()
# Convert pd dataFrame to np array
data_array = data.values


# READING FROM PICKLE FILES



# READING FROM EXCEL FILES
import pandas as pd
file = 'analyticsFile.xlsx'
XL = pd.ExcelFile(file)
print(XL.sheet_names)
print(XL.keys()) # Prints the sheet names also
# Read specific sheet
df1 = XL.parse('Sheet1') # Use sheet 1 of excel
df2 = XL.parse('Sheet2')
print(df2.head())
df3 = XL.parse(0, skiprows=[0], names=['Country','Songs']) # Parses first sheet, skips 1st row, and renames columns
df4 = XL.parse(1, parse_cols=[0], skiprows=[0], names=['Country']) # Parses second sheet, only first column, skips 1st row, and renames columns
# To read all sheets... Use None for SheetName
XL = pd.read_excel(file, sheetname = None)


# READING FROM STATA FILE
import pandas as pd
df = pd.read_stata('filename.dta')


# READING FROM HDF5 FILES - large numerical data
import h5py
filename = 'H-H1_LOSC_dsaflk_jdkf.hdf5'
data = h5py.File(filename, 'r')
print(type(data))
# Loop for getting meta data names
for key in data.keys():
    print(key)


# READING FROM MATLAB
# SciPy needs to be used
# .mat files are collection of variables in a workspace
import scipy.io
filename = 'workspace.mat'
mat = scipy.io.loadmat(filename)
print(type(mat))


# READING FROM RELATIONAL DATABASE
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite') #Northwind is the name of the DB in sqlite
table_names = engine.table_names() # Get the table names to understand the SQL structure
print(table_names)
# Create connection
con = engine.connect()
rs = con.execute("SELECT * FROM Names")
df = pd.DataFrame(rs.fetchall()) # Conver to pandas DF
df.columns = rs.keys() # Renaming the columns to the key names
con.close() # Close the connection
print(df.head())

# We can get more specific with this...
engine = create_engine('sqlite:///Northwind.sqlite') #Northwind is the name of the DB in sqlite
with engine.connect() as con:
    rs = con.execute("SELECT OrderID, ORderDate, ShipName FROM Orders")
    df = pd.DataFrame(rs.fetchmany(size = 5))
    df.columns = rs.keys() # Renaming the columns to the key names
# We can do this even easier with PANDAS!
df = pd.read_sql_query("SELECT * FROM Orders", engine)

#########################################
######################################### READING FROM THE WEB
######################################### SCRAPING

# URLlib package accepts URLs instead of filenames
from urllib.request import urlretrieve
import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
urlretrieve(url, 'winequality-white.csv') # Saves file locally-- THIS STEP ISNT NECESSARY
df = pd.read_csv('winequality-white.csv', sep = ';')
print(df.head())

# GET requests using URLlib
from urllib.request import urlopen, Request
url = "https://www.wikipedia.org/"
request = Request(url)
response = urlopen(request)
html = response.read()
print(html)
response.close()
# OR USE REQUESTS PACKAGE FOR API-- GET REQUESTS USING REQUESTS --this is HIGHER level
import requests
url = "https://www.wikipedia.org/"
r = requests.get(url)
text = r.text

# USE BEAUTIFULsoup for scraping easier
# TURNING A WEBPAGE INTO DATA USING BEAUTIFULsoup
from bs4 import BeautifulSoup
import requests
url = 'https://www.crummy.com/software/BEAUTIFULsoup/'
r = requests.get(url) # Package the request, send the request, and get response
html.doc = r.text # Extracts the response as html
soup = BeautifulSoup(html.doc)
print(soup.prettify())
print(soup.title) # Gets title
print(soup.get_text()) # Gets text
find_all() # Extracts all the hyperlinks on the page
# Gets all hyperlinks on a page
a_tags = soup.find_all('a') # <a> defines hyperlinks in HTML but we dont use the curve brackets with find_all
for link in a_tags:
    print(link.get('href')) # Will print all hyperlinks nicely

# USING APIs and JSONs
# JSONs are typical way to get data from APIs
# We should always store the info from JSON into a dict in Python
import json
with open('snakes.json', 'r') as json_file:
    json_data = json.load(json_file)
type(json_data) # should show dict for dictionary
for key, value in json_data.items():
    print(key + ':', value) # prints key:value pairs in dictionary
# Could also do this loop instead...
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])

# API: Set of protocols and routines for interacting with software apps
import requests
ur = 'http://www.omdbapi.com/?t=hackers'
r = requests.get(url)
json_data = r.json() # Decode the JSON into a dict
for key, value in json_data.items():
    print(key + ':', value)

# TWITTER API
# Use Tweety package for noobs
import tweepy, json
access_token = "70525371-Xgx7vYcSnNHfdwTlhosUCH2imdHLy5fFykRS8o40y"
access_token_secret = "HJddmGJJiSInaGq0o8K5LbAX1ciIg4Nrd90IKHakHUlyN"
consumer_key = "MN6zKaQ1rKd8uQnQShYWSbQQu"
consumer_secret = "WjTawG61qAHqePNq8UzyR4XKdqvD7DKxJJtLgo0xAlCIdFPrJR"

auth = tweep.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
# Create a listener of Twitter data since it's streaming
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open("tweets.txt", "w")

    def on_status(self, status):
        tweet = status._json
        self.file.write( json.dumps(tweet) + '\n' )
        self.num_tweets += 1
        if self.num_tweets < 100:
            return True
        else:
            return False
        self.file.close()

    def on_error(self, status):
        print(status)
l = MyStreamListener()
stream = tweepy.Stream(auth, l)
stream.filter(track=['clinton', 'trump', 'sanders', 'cruz'])
import json
# String of path to file: tweets_data_path
tweets_data_path = 'tweets.txt'
# Initialize empty list to store tweets: tweets_data
tweets_data = []
# Open connection to file
tweets_file = open(tweets_data_path, "r")
# Read in tweets and store in list: tweets_data
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)
# Close connection to file
tweets_file.close()
import pandas as pd
df = pd.DataFrame(tweets_data, columns = ['text','lang'])
print(df.head())
# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]
# Iterate through df, counting the number of tweets in which
# each candidate is mentioned
for index, row in df.iterrows():
    clinton += word_in_text('clinton', row['text'])
    trump += word_in_text('trump', row['text'])
    sanders += word_in_text('sanders', row['text'])
    cruz += word_in_text('cruz', row['text'])
# Import packages
import matplotlib.pyplot as plt, seaborn as sns
# Set seaborn style
sns.set(color_codes=True)
# Create a list of labels:cd
cd = ['clinton', 'trump', 'sanders', 'cruz']
# Plot histogram
ax = sns.barplot(cd, [clinton, trump, sanders, cruz])
ax.set(ylabel="count")
plt.show()
