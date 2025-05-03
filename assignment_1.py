import pandas as pd

dataset = pd.read_csv('Walmart.csv')

print("Dataset Loaded")
# print the csv file
# print(dataset) 
# print(dataset.describe) 

#get first 5 rows
print(dataset.head()) 


#get last 10 rows
print(dataset.tail(10)) 

#get values : get all rows 
print(dataset.values) 


#get size
print(dataset.shape) 


#get dataype of each column 
print(dataset.dtypes) 


#print values : get all rows 
print(dataset.values) 


#get columns name
print(dataset.columns)

#set csv one column as index
# dataset1 = pd.read_csv('Walmart.csv', index_col="Date")
# print(dataset1) 

#get total weekly_sales
# print(dataset.Weekly_Sales)  # print single column
print(f'Total weekly sales = {dataset.Weekly_Sales.sum()}') 

#copy a column 
weekly_sales  = dataset["Weekly_Sales"].copy()
print(weekly_sales) 

# get some columns from csv data
cols = ['Date', 'Weekly_Sales',  'Fuel_Price']
print(dataset[cols])

# insert a new column
print(dataset.head())
dataset.insert(loc=4, column="Description", value="describe")
print(dataset.head())

# drop rows that has missing value
print(dataset.dropna(how="any")) # any of the row missing value
print(dataset.dropna(how="all")) # all the row missing value
print(dataset.dropna(subset=["Fuel_Price"]))
print(dataset.dropna(subset=["Fuel_Price", "Temperature"])) # any of the two column has missing value

# fill missing values
print(dataset.fillna(0)) # fill all the missing values with 0
print(dataset["Fuel_Price"].fillna(0)) # fill specfic column with 0 

print("-------------------------------")
# find the highest temperature
max_temperature =dataset["Temperature"].max()
print(f"Highest Temperate : { max_temperature}") 

# find the lowest gas price 
min_price =dataset["Fuel_Price"].min()
print(f"Lowest gas price : { min_price}")