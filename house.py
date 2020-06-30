#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing dataset
data = pd.read_csv("train.csv")

#Analysing data for relevant columns

desc = data.describe()
print(desc)
data.columns

desc['SalePrice'].describe
desc['YearBuilt'].describe
desc['GrLivArea'].describe
desc['TotalBsmtSF'].describe
desc['OverallQual'].describe

#Creating distplot for saleprice
sns.distplot(data['SalePrice'])

#Creating histogram for SalePrice
plt.hist(data['SalePrice'])

#Skewness and Kurtosis
print("Skewness: %f" % data['SalePrice'].skew())
print("Kurtosis: %f" % data['SalePrice'].kurt())

#Creating distplot for TotalBsmtSF
sns.distplot(data['TotalBsmtSF'])

#Creating histogram for TotalBsmtSF
plt.hist(data['TotalBsmtSF'])

#Creating distplot for GrLivArea
sns.distplot(data['GrLivArea'])

#Creating histogram for GrLivArea
plt.hist(data['GrLivArea'])

#Creating distplot for YearBuilt
sns.distplot(data['YearBuilt'])

#Creating histogram for YearBuilt
plt.hist(data['YearBuilt'])

#Creating distplot for OverallQual
sns.distplot(data['OverallQual'])

#Creating histogram for OverallQual
plt.hist(data['OverallQual'])

#Creating scatter plot for SalePrice and TotalBsmtSF
plt.scatter(data['TotalBsmtSF'], data['SalePrice'])
plt.show()

#Creating scatterplot for SalePrice and GrLivArea
plt.scatter(data['GrLivArea'], data['SalePrice'])
plt.show()

#Creating scatterplot for SalePrice and YearBuilt
plt.scatter(data['YearBuilt'], data['SalePrice'])
plt.show()

#Creating a boxplot fpr YearBuilt and SalePrice
sns.boxplot(x = data['YearBuilt'], y = data['SalePrice'], data = data)

#Creating a pairplot for less relevant variables as well
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(data[cols], size = 2.5)
plt.show();