#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#1.Project Background

#Lending Club connects people who need money (borrowers) with people who have money (investors).
#Try to create a model that will help predict people who have a profile of having a high probability of paying back.
#Lending club had a very interesting year in 2016.
#This data is from before they even went public.


# In[ ]:


# 2. Objective 
#We are using lending data from 2007-2010 to classify and predict whether or not the borrower paid back their loan in full.


# In[72]:


# 3. Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report                   # To generate classification report
from sklearn.metrics import plot_confusion_matrix 
from sklearn.model_selection import GridSearchCV

import warnings 
warnings.filterwarnings("ignore")


# In[ ]:


# 4. Data Acquisition and description

#4.1 Data description

#Column Name	Description
#credit.policy	1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.

#purpose	The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", 
#"major_purchase", "small_business", and "all_other").

#int.rate	The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). 
#Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.

#installment	The monthly installments owed by the borrower if the loan is funded.

#log.annual.inc	The natural log of the self-reported annual income of the borrower.

#dti	The debt-to-income ratio of the borrower (amount of debt divided by annual income).

#fico	The FICO credit score of the borrower.

#days.with.cr.line	The number of days the borrower has had a credit line.

#revol.bal	The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).

#revol.util	The borrower's revolving line utilization rate (the amount of the credit line used relative to 
#total credit available).

#inq.last.6mths	The borrower's number of inquiries by creditors in the last 6 months.

#delinq.2yrs	The number of times the borrower had been 30+ days past due on a payment in the past 2 years.

#pub.rec	The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

#not.fully.paid ------> The borrower did not fully pay the credit=1;otherwise=0


# In[8]:


#4.2 Data acquisition

loans= pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-2/master/Data/loan_data.csv')
loans.head()


# In[10]:


loans.describe()


# In[11]:


loans.columns


# In[12]:


loans.shape


# In[14]:


loans.info()

#The data has no missing values


# In[ ]:


#5.Exploratory Data Analysis


# In[16]:


#5.1 Making a pie chart for variable credit.policy; 1 depicts customer meets criteria;0 depicts customer does not meets criteria

# Creating a custom figure size of 10 X 10 inches
figure = plt.figure(figsize=[10,10])


# Using magic of pandas pie() function
loans['credit.policy'].value_counts().plot(kind='pie', fontsize=14,
                                      autopct='%3.1f%%', wedgeprops=dict(width=0.15),
                                      shadow=True, startangle=160, cmap='inferno', legend=True)
plt.ylabel(ylabel='credit.policy',size=14)
plt.title(label='Donut plot showing the proportion of each category value', size=16)
plt.show()

#So 80% customers met crietria for credit policy


# In[17]:


#5.2 Bar chart for 'purpose' of loan

# Creating a custom figure size of 15 X 7 inches
figure = plt.figure(figsize=[15,7])

# Using magic of pandas bar function
(loans['purpose'].value_counts()/len(loans)).plot.bar(color='#3F617A')

# Changing x-ticks label size to 12 and rotating to 90 degrees
plt.xticks(rotation=90, size=12)

# Changing y-ticks value using an array of size 0.3 with a step size of 0.05
plt.yticks(ticks=np.arange(0, 0.3, 0.5), size=12)

# Labelling x-axis with a custom label and size of 14
plt.xlabel(xlabel='Purpose of taking loan', size=14)

# Labelling y-axis with a custom label and size of 14
plt.ylabel(ylabel='Reasons', size=14)

#labelling title with a custom label and size of 16
plt.title(label='Bar chart showcasing purposes of customers taking loan', size=16)

#display the output by rendering visuals on the screen
plt.show()

#Majority of customers took loan for debt consolidation followed by others.


# In[25]:


#5.3 Interpreting interest rate of loan, which is being on higher side for borrowers considered more risky

loans['int.rate'].describe()

#The mean interest rate being charged was 12%, with the variation of maximum being 21.6% and minimum being 0.6%.


# In[26]:


#5.4 Interpreting 'installment' on monthly basis being owned by borrower

loans['installment'].describe()


# In[27]:


#5.5 Interpreting 'log.annual.inc' i.e. the natural log of self-reported annual income of borrower

loans['log.annual.inc'].describe()


# In[28]:


#5.6 Interpreting 'dti' i.e. debt to income ratio of the borrower

loans['dti'].describe()

#The debt to income ratio was on an average 12% with minimum being 0 and maximum being 30%. 
#More than three fourth of the customers had debt to income ratio of 18%.


# In[29]:


#5.7 Interpreting 'fico' i.e. the FICO credit score of the borrower

loans['fico'].describe()

#The mean FICO score was 710 with a standard deviation of 38.


# In[41]:


#Plotting a histogram of two FICO distribution on top of each other, one for each credit.policy
import matplotlib.pyplot as plt
def tree():
    plt.figure(figsize=(10,6))
    loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
    loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
    plt.legend()
    plt.xlabel('FICO')
tree()


# In[30]:


# 5.8 Interpreting 'days.with.cr.line'

loans['days.with.cr.line'].describe()

# The mean number of days the borrowers had credit line were 4560 with a standard deviation of 2496.


# In[31]:


#5.9 Interpreting 'revol.bal' i.e. the unpaid amount at the end of credit card billing cycle

loans['revol.bal'].describe()


# In[32]:


#5.10 Interpreting 'revol.util' i.e. amount of credit line used relative to totall credit available

loans['revol.util'].describe()


# In[33]:


#5.11 Interpreting 'inq.last.6mths' i.e. borrower's no. of queries by creditors in last 6 months

loans['inq.last.6mths'].describe()

# Three quater of the creditors did not have more than 2 queries from borrowers in last 6 months


# In[34]:


# 5.12 Interpreting 'delinq.2yrs' i.e. no. of times borrower had past 30 days post due date for payment
#in last two years

loans['delinq.2yrs'].describe()


# In[35]:


#5.13 Interpreting 'pub.rec' i.e. borrowere derogatory public records

loans['pub.rec'].describe()


# In[42]:


#5.14 Plot a histogram of two FICO distribution on top of each other, one for each not.fully.paid

import matplotlib.pyplot as plt
def tree():
    plt.figure(figsize=(10,6))
    loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
    loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
    plt.legend()
    plt.xlabel('FICO')
tree()


# In[43]:


#Using seaborn countplot showing the counts of loans by purpose, with the color hue defined by not.fully.paid

import seaborn as sns
def tree():
    plt.figure(figsize=(11,7))
    sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
tree()


# In[ ]:


#6. Post Data Processing


# In[ ]:



#6.1 Feature Selection
#Here we will visualize  the correlation of input features using Heatmap.
#If we see a case of correlation we will remove the highly correlated feature.


# In[36]:


loans.corr()


# In[39]:


cor = loans.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cor, cmap='YlGnBu', annot=True, square=True)
plt.show()


# In[ ]:


#6.2 Data Encoding

#6.2 Data Encoding
#In this section, we will encode our categorical features such as Sex, Embarked, Title using one hot encoding.

#Hot encoding is a technique that we use to represent categorical variables as numerical values in a machine learning model. 
#The advantages of using one hot encoding include: It allows the use of categorical variables in models that 
#require numerical input.


# In[47]:


loans = pd.get_dummies(data=loans, columns=['purpose'])
loans.head()


# In[ ]:


#6.3 Data Preparation
#No data scaling is required for Decision Tree as they are giant if-else conditional statements.

#Spliting of data into dependent and independent variables for further development


# In[48]:


X = loans.loc[ : ,loans.columns!='not.fully.paid']
X.head()


# In[52]:


y=loans.loc[ : ,loans.columns=='not.fully.paid']
y.head()


# In[54]:


#6.4 Splitting data into test and train dataset
#Splitting data into testing and training dataset

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42, stratify=y)


#Display the shape of training and testing data
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


# In[ ]:


#7.Model Developement and Evaluation

#In this section we will develop Decision Tree model
#Then we will analyze the results obtained and make our observations.
#We will use several parameteres to see how the model can be best fit with using different indexes


# In[57]:


#7.1 Fitting the model -Basemodel
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train,y_train)
model


# In[58]:


#7.1.2 Using model for prediction
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)


# In[61]:


#7.1.3 Plotting confusion matrix of train and test data

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(15, 7))
plot_confusion_matrix(estimator=model, X=X_train, y_true=y_train, values_format='.5g', cmap='YlGnBu', ax=ax1)
plot_confusion_matrix(estimator=model, X=X_test, y_true=y_test, values_format='.5g', cmap='YlGnBu', ax=ax2)
ax1.set_title(label='Train Data', size=14)
ax2.set_title(label='Test Data', size=14)
ax1.grid(b=False)
ax2.grid(b=False)
plt.suptitle(t='Confusion Matrix', size=16)
plt.show()


# In[62]:


train_report = classification_report(y_train, y_pred_train)
test_report = classification_report(y_test, y_pred_test)

print('                Training Report              ')
print(train_report)
print('                Testing Report               ')
print(test_report)

#model is overfitted as accuracy of train data in 1 and that of test data is 0.74


# In[66]:


#7.2 Model 2 by using criteria like gini or entropy, max_depth, min_samples_split and min_samples_leaf

hp_to_be_tuned = {
    "criterion" : ["gini","entropy"],
    "max_depth" : [1,2,3,4,5,6,7,None],
    "min_samples_split" : [1,2,3,4,5,6,7,None],
    "min_samples_leaf" : [1,2,3,4,5,6,7,None]
}


# In[78]:


model2 = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=10, max_depth=3,min_samples_split=5,min_samples_leaf=4)
     


# In[79]:


#7.2.1 Fitting the model
model2.fit(X_train, y_train)


# In[80]:


#7.2.3 Use the model for prediction
y_pred_test = model2.predict(X_test)
y_pred_train = model2.predict(X_train)


# In[ ]:


# 7.2.4 Model Evaluation using confusion matrix and report


# In[81]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(15, 7))
plot_confusion_matrix(estimator=model2, X=X_train, y_true=y_train, values_format='.5g', cmap='YlGnBu', ax=ax1)
plot_confusion_matrix(estimator=model2, X=X_test, y_true=y_test, values_format='.5g', cmap='YlGnBu', ax=ax2)
ax1.set_title(label='Train Data', size=14)
ax2.set_title(label='Test Data', size=14)
ax1.grid(b=False)
ax2.grid(b=False)
plt.suptitle(t='Confusion Matrix', size=16)
plt.show()


# In[82]:


train_report = classification_report(y_train, y_pred_train)
test_report = classification_report(y_test, y_pred_test)

print('                Training Report              ')
print(train_report)
print('                Testing Report               ')
print(test_report)

#The difference between the training and test accuracy score is nil.


# In[ ]:


#8. Conclusion
#After modelling the accuracy score of train and test data almost matches. But the predict accurately we need
#statistical score of AUC

