
``````````
# ITERATING

enumerate()
zip()
iter()
next()

days = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
days1 = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# user iter() to create an iteration over collection
i = iter(days)
print(next(i))
print(next(i))
print(next(i))

# Cool but not super helpful

# This might be better
for m in days:
	print(m)

# This would give day, index
for m in range(len(days)):
	print(m, days[m])
# Enurmerate would be better here...

for i, m in enumerate(days, start=1):
	print(i, m)
# Much better!

# Zip can combine sequences
for m in zip(days, days1):
	print(m)

# Zip and enumerate are very powerful together
for i, m in enumerate(zip(days, days1), start=1):
	print(i, m[0], "=", m[1], "is offset by 1")


``````````
# TRANSFORMING DATA

sorted()
filter()
map()

nums = (1,5,8,4,43,34,410,59,47)
chars = 'abcDFEFGijjlJMO'
grades = (81, 89, 94, 78, 83, 55, 74)

# Filter out even numbers
def filterFunc(x):
	if x % 2 == 0:
		return False
	return True

odds = list(filter(filterFunc, nums))
print(odds)

# Filter for lower chars
def filterFunc2(x):
	if x.isupper():
		return False
	return True

loweres = list(filter(filtFunc2, chars))
print(lowers)

# Map to creat sequence of values
def squareFunc(x):
	return x ** 2

squares = list(map(squareFunc, nums))
print(squares)

# Sort and map number grades to letter grades
def toGrade(x):
	if (x >= 90):
		return "A"
	elif (x >= 80 and x < 90):
		return "B"
	elif (x >= 70 and x < 80):
		return "C"
	elif (x >= 60 and x < 70):
		return "D"
	return "F"

grades = sorted(grades)
letters = list(map(toGrade, grades))
print(letters)


``````````
# ITER TOOLS Modele

import itertools
seq1 = ["Joe","John","Mike"]

# Cycles over set of values infitnitely
cycle1 = itertools.cycle(seq1)
print(next(cycle1))
print(next(cycle1))
print(next(cycle1))
print(next(cycle1))

# Count Iterator
count1 = itertools.count(100, 10) # Start and Step
print(next(count1))
print(next(count1))
print(next(count1))

# Accumulatet
vals = [10, 20, 30, 40, 50, 60, 40, 30, 20]
acc = itertools.accumulate(vals)
print(acc) # Running addition is default-- Can change to max or all different functions

# Chain Function
x = itertools.chain("ABCD", "1234")
print(list(x))
# one list of values from separate lists

# Drop While / Take While
def testFunction(x):
	return x < 40

print(list(itertools.dropwhile(testFunction, vals))) # Will drop values from sequence while test function returns True
print(list(itertools.takewhile(testFunction, vals))) 