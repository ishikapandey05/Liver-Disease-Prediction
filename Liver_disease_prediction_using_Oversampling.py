#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')

liver_data = pd.read_csv("C:/Users/Dell/anaconda3/Lib/site-packages/pandas/core/data/indian_liver_patient.csv")


# In[2]:


# Success - Display the first record
display(liver_data.head(n=5))
# liver_data.head(n=5)


# In[3]:


liver_data.info()


# In[4]:


liver_data.columns


# In[5]:


liver_data.Dataset.value_counts()


# In[6]:


import seaborn as sns

n_records = len(liver_data.index)
n_records_liv_pos = len(liver_data[liver_data['Dataset'] == 1])
n_records_liv_neg = len(liver_data[liver_data['Dataset'] == 2])
percent_liver_disease_pos = (n_records_liv_pos/n_records)*100

print("Number of records: {}".format(n_records))
print("Number of patients likely to have liver disease {}".format(n_records_liv_pos))
print("Number of patients unlikely to have liver disease {}".format(n_records_liv_neg))
print("Percentage of patients likely to have liver disease {}%".format(percent_liver_disease_pos))

sns.countplot(data=liver_data, x = 'Dataset', label='Count')


# In[7]:


liver_data_labels = liver_data['Dataset']
# Drop label feature
liver_data_features = liver_data.drop(['Dataset'], axis=1)
# liver_data_features.head()


# In[8]:


#Missing values
display(liver_data_features[liver_data_features['Albumin_and_Globulin_Ratio'].isnull()])


# In[9]:


# fill missing values with median value
liver_data_features.Albumin_and_Globulin_Ratio.fillna(liver_data_features['Albumin_and_Globulin_Ratio'].median(), inplace=True)

albumin_globulin_missing_indices = [209, 241, 253, 312]
liver_data_features.loc[albumin_globulin_missing_indices].head()


# In[10]:


# plot features histogram
liver_data_features.hist(figsize=(14,10))


# In[11]:


# Having a look at the correlation matrix

fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(liver_data.corr(numeric_only=True), annot=True, fmt='.1g', cmap="viridis", cbar=False);


# In[12]:


import seaborn as sns
sns.set_style("whitegrid")  # Sets the seaborn style to "whitegrid"

fig, ax = plt.subplots(figsize=(4,4))

plt.pie(x=liver_data["Dataset"].value_counts(), 
        colors=["firebrick","seagreen"], 
        labels=["UnHealthy Liver","Healthy Liver"], 
        shadow = True, 
        explode = (0, 0.1)
        )

plt.show()


# In[13]:


skewed = ['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Albumin_and_Globulin_Ratio']

liver_data_features_log_transformed = pd.DataFrame(data = liver_data_features)
liver_data_features_log_transformed[skewed] = liver_data_features[skewed].apply(lambda x: np.log(x))

liver_data_features_log_transformed.hist(figsize=(14,10))


# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# calculate correlation coefficients for the dataset
correlations = liver_data.corr(numeric_only=True)

# and visualize
plt.figure(figsize=(10, 10))
g = sns.heatmap(correlations, cbar = True, square = True, annot=True, fmt= '.2f', annot_kws={'size': 10})


# In[15]:


sns.jointplot(x="Total_Bilirubin",y="Direct_Bilirubin", data=liver_data, kind="reg",height=4)
sns.jointplot(x="Alamine_Aminotransferase",y="Aspartate_Aminotransferase", data=liver_data, kind="reg",height=4 )


# In[16]:


sns.jointplot(x="Total_Protiens",y="Albumin", data=liver_data, kind="reg",height=4)
sns.jointplot(x="Albumin",y="Albumin_and_Globulin_Ratio", data=liver_data, kind="reg",height=4) 


# ### Based on the correlation plots, the following pairs of features seem to be related:1. 
# Total_Bilirubin & Direct_Bilirubi
# 2. 
# Alamine_Aminotransferase & Aspartate_Aminotransfera
# 3. e
# Total_Protiens & Albu
# 4. in
# Albumin & Albumin_and_Globulin_R
# atio

# In[17]:


# Independent and Dependent Feature:
X = liver_data.iloc[:, :-1]
y = liver_data.iloc[:, -1]



# In[18]:


X = liver_data.drop(columns='Dataset', axis=1)
y = liver_data['Dataset']


# In[19]:


X


# In[20]:


y


# In[21]:


# Convert the Gender column to a string data type
liver_data['Gender'] = liver_data['Gender'].astype(str)

# Encode the Gender column with 1 for Male and 0 for Female
liver_data['Gender'] = np.where(liver_data['Gender']=='Male', 1, 0)

# Convert the Gender column to a numeric data type
liver_data['Gender'] = liver_data['Gender'].astype(int)


# In[22]:


display(liver_data)


# In[23]:


from imblearn.over_sampling import RandomOverSampler
# Create a RandomOverSampler object
oversampler = RandomOverSampler()

# Fit the oversampler to the data
oversampler.fit(X, y)

# Resample the data
X_resampled, y_resampled = oversampler.fit_resample(X, y)
print("X_resampled shape:", X_resampled.shape)
print("Y_resampled shape:", y_resampled.shape)


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, y_resampled, test_size=0.35, random_state=101)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ## Training the model using Logistic Regression

# In[25]:


from sklearn.preprocessing import OneHotEncoder

# Assuming 'X_train' and 'X_test' contain your feature variables, including categorical ones

# Create an instance of OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform the categorical columns in X_train
X_train_encoded = encoder.fit_transform(X_train)

# Transform the categorical columns in X_test (using the fitted encoder)
X_test_encoded = encoder.transform(X_test)


# In[26]:


from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(max_iter=1000) #Increasing number of iterations for logistic Regression
model1.fit(X_train_encoded, Y_train)


# In[27]:


# accuracy on training data
from sklearn.metrics import accuracy_score
X_train_prediction = model1.predict(X_train_encoded)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data using Logistic Regression : ', training_data_accuracy*100)


# In[28]:


# accuracy on test data
X_test_prediction = model1.predict(X_test_encoded)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data using Logistic Regression : ', test_data_accuracy*100)


# ## Training the model using Random Forest Classifier

# In[29]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators = 100)
model2.fit(X_train_encoded,Y_train)


# In[30]:


X_train_prediction2 = model2.predict(X_train_encoded)
training_data_accuracy2 = accuracy_score(X_train_prediction2, Y_train)
print('Accuracy on Training data using Random Forest Classifier : ', training_data_accuracy2*100)


# In[31]:


X_test_prediction2 = model2.predict(X_test_encoded)
test_data_accuracy2 = accuracy_score(X_test_prediction2, Y_test)
print('Accuracy on Test data using Random Forest Classifier : ', test_data_accuracy2*100)


# ## Training the model using Decision Tree Classifier

# In[32]:


from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier(random_state=101)
model3.fit(X_train_encoded, Y_train)


# In[33]:


X_train_prediction3 = model3.predict(X_train_encoded)
training_data_accuracy3 = accuracy_score(X_train_prediction3, Y_train)
print('Accuracy on Training data using Decision Tree Classifier : ', training_data_accuracy3*100)


# In[34]:


X_test_prediction3 = model3.predict(X_test_encoded)
test_data_accuracy3 = accuracy_score(X_test_prediction3, Y_test)
print('Accuracy on Test data using Decision Tree Classifier : ', test_data_accuracy3*100)


# ## Creating Data Frame for the accuracy score of logistic regression,Random Forest Classifier and Decision Tree Classifier

# In[35]:


import pandas as pd

# Create DataFrames for each model's results
results_data = [
    ["Logistic Regression", training_data_accuracy*100, test_data_accuracy*100],
    ["Random Forest Classifier", training_data_accuracy2*100, test_data_accuracy2*100],
    ["Decision Tree Classifier", training_data_accuracy3*100, test_data_accuracy3*100]
]

# Create a list of DataFrames
results_dfs = [pd.DataFrame(data=[row], columns=['Model', 'Training Accuracy %', 'Testing Accuracy %']) for row in results_data]

# Concatenate the DataFrames
results_df = pd.concat(results_dfs, ignore_index=True)

# Display the concatenated DataFrame
print(results_df)


# ## So by Analysing the above table we can easily guess that Random Forest Classifier is performing well for Training and Testing accuracy score. So we will Continue Predicting the new Data with Random Forest Classifier Model.

# In[ ]:




