#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats

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

#Solving for normality and homoscedasticity fro relevant variables

#Normality

#SalePrice
pt = stats.probplot(data['SalePrice'], plot=plt)
#Applying log transformation
data['SalePrice'] = np.log(data['SalePrice'])
sns.distplot(data['SalePrice'])
pt = stats.probplot(data['SalePrice'], plot=plt)

#GrLivArea
pt = stats.probplot(data['GrLivArea'], plot=plt)
#Applying log transformation
data['GrLivArea'] = np.log(data['GrLivArea'])
sns.distplot(data['GrLivArea'])
pt = stats.probplot(data['GrLivArea'], plot=plt)

#TotalBsmtSF
sns.distplot(data['TotalBsmtSF'])
pt = stats.probplot(data['TotalBsmtSF'], plot=plt)

#Log transformation not possible directly
#Use categorical variables
#if area>0 it gets 1, for area==0 it gets 0
data['BsmtPresent'] = pd.Series(len(data['TotalBsmtSF']), index=data.index)
data['BsmtPresent'] = 0 
data.loc[data['TotalBsmtSF']>0,'BsmtPresent'] = 1

#Log transformation
data.loc[data['BsmtPresent']==1,'TotalBsmtSF'] = np.log(data['TotalBsmtSF'])

#Visualization
sns.distplot(data[data['TotalBsmtSF']>0]['TotalBsmtSF']);
res = stats.probplot(data[data['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


#Homoscedasticity
plt.scatter(data['GrLivArea'], data['SalePrice'])
plt.scatter(data[data['TotalBsmtSF']>0]['TotalBsmtSF'], data[data['TotalBsmtSF']>0]['SalePrice'])


#Converting categorical variable to dummy variable
data = pd.get_dummies(data)
