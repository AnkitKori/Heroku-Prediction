# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# for warning
from warnings import filterwarnings
filterwarnings("ignore")  ## To remove any kind of warning

#Reading the dataset
data = pd.read_csv('heart.csv')
data.head()

# Show rows and columns
data.shape

#Info
data.info()

#Checking the unique values for each column
data.nunique()

#Statistical measures
data.describe()

#Checking the missing values
data.isnull().sum()

#Checking the distributions of Target variable
data['target'].value_counts()

#Exploratory Data Analysis
sns.set_style("whitegrid")
sns.countplot(x='target',data=data,palette='RdBu_r')

# Correlation matrix
corr = data.corr()
plt.subplots(figsize=(20,20))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap="YlGnBu")

## Pairplotting of dataframe
import seaborn as sns
numeric_data = data[['age','trestbps','chol','thalach','oldpeak']]
sns.pairplot(numeric_data)

#Histogram
data.hist(figsize=(14,14))
plt.show()

# create four distplots
plt.figure(figsize=(12,10))
plt.subplot(221)
sns.distplot(data[data['target']==0].age)
plt.title('Age Of Patients Without Heart Disease')

plt.subplot(222)
sns.distplot(data[data['target']==1].age)
plt.title('Age Of Patients With Heart Disease')

# PREPARE DATA FOR MODELLING
X = data.drop(['target'],axis = 1)
y = data['target']

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 2)

from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


model1 = LogisticRegression() # get instance of model
model1.fit(x_train, y_train) # Train/Fit model 

y_pred1 = model1.predict(x_test)
print(classification_report(y_test, y_pred1)) 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred1)
print(confusion_matrix)

# Saving model to disk
pickle.dump(model1, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))