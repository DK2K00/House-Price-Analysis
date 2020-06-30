#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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

#Heatmap to check correlation of SalePrice
n = 10 #number of variables
corrmat = data.corr()
col = corrmat.nlargest(n, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data[col].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=col.values, xticklabels=col.values)
plt.show()

#Data Manipulation

#Checking amount of missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)
missing_data.tail(10)

#Removing ad dealing with missing data
data = data.drop((missing_data[missing_data['Total'] > 1]).index,1)
data = data.drop(data.loc[data['Electrical'].isnull()].index)
data.isnull().sum().max()

#Dealing with outliers

#Deleting outlier points for GrLivArea
data.sort_values(by = 'GrLivArea', ascending = False)[:2]
data = data.drop(data[data['Id'] == 1299].index)
data = data.drop(data[data['Id'] == 524].index)

#standardizing data
saleprice_scaled = StandardScaler().fit_transform(data['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('Outside of lower range of the distribution:')
print(low_range)
print('\nOutside of higher range of the distribution:')
print(high_range)