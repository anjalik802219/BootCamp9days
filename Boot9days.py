#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


# In[24]:


df  = pd.read_csv("c:\\Users\\Anjali Kumari\\OneDrive\\Desktop\\archive (2)\\StressLevelDataset.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.columns


# In[7]:


df.tail


# In[8]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[10]:


df_encoded = pd.get_dummies(df, drop_first=True)


# In[11]:


correlation_matrix_encoded = df_encoded.corr()
correlation_matrix_encoded


# In[12]:


from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"Accuracy: {rf.score(X_test, y_test):}")


# In[13]:


import xgboost as xgb
from sklearn.metrics import accuracy_score  # Corrected the import

# Prepare the data in DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)  # Corrected to DMatrix
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100
}

# Train the XGBoost model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# Predict using the trained model
y_pred = xgb_model.predict(dtest)

# Convert probabilities to binary predictions
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)  # Corrected y_tset to y_test
print(f"XGBoost Accuracy: {accuracy}")


# In[14]:


from sklearn.ensemble import AdaBoostClassifier
#Intialize AdaBoost
ada = AdaBoostClassifier(n_estimators=50,random_state=42)
ada.fit(X_train,y_train)

# Predict and Evaluate
y_pred = ada.predict(X_test)
print(f"AdaBoost Accuracy:{ada.score(X_test,y_test)}")


# In[15]:


import numpy as np

# Simple dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 23])

# Initialize parameters
m = 0
b = 0
learning_rate = 0.01  # Corrected the spelling of learning_rate
epochs = 1000

# Gradient Descent Algorithm
for _ in range(epochs):
    y_pred = m * X + b
    D_m = (-2 / len(X)) * sum(X * (y - y_pred))  # Added missing parenthesis
    D_b = (-2 / len(X)) * sum(y - y_pred)
    m -= learning_rate * D_m
    b -= learning_rate * D_b

# Final slope and intercept
print(f"Slope: {m}, Intercept: {b}")


# In[16]:


sns.histplot(df['self_esteem'], kde=True,color='skyblue')
plt.title('Distribution of self_esteem ')
plt.show()


# In[17]:


sns.histplot(df['depression'], kde=True, color='green')
plt.title('Distribution of depression')
plt.show()


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

num_cols = len(df.columns)
plt.figure(figsize=(20, 5 * ((num_cols // 3) + 1)))
for i, col in enumerate(df.columns, 1):
    plt.subplot((num_cols // 3) + 1, 3, i)  
    sns.histplot(x=df[col])
    plt.title(f"Histogram of {col} Data")
    plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# In[19]:


df_corr = df.corr()
plt.figure(figsize = (12, 12))
sns.heatmap(df_corr, fmt = ".3f", annot = True, cmap = "pink")
plt.show()


# In[20]:


sns.scatterplot(x = df["blood_pressure"], y = df["blood_pressure"])
plt.show()


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def cat_analyser(data, col: str, freq_limit: int = 36):
    df_ = data.copy()
    sns.set(rc={'axes.facecolor': 'gainsboro', 'figure.facecolor': 'gainsboro'})
    
    if df_[col].nunique() > freq_limit:
        df_ = df_.loc[df_[col].isin(df_[col].value_counts().keys()[:freq_limit].tolist())]
    
    # Create subplots for countplot and pie chart
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    fig.suptitle(col, fontsize=16)
    
    # Countplot
    sns.countplot(data=df_, x=col, ax=ax[0], palette='coolwarm', 
                  order=df_[col].value_counts().index)
    ax[0].set_xlabel('')

    # Pie chart
    pie_cmap = plt.get_cmap('coolwarm')
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    df_[col].value_counts().plot.pie(autopct='%1.1f%%', textprops={'fontsize': 10},
                                      ax=ax[1], colors=pie_cmap(normalize(df_[col].value_counts())))
    ax[1].set_ylabel('')
    
    plt.show()
    matplotlib.rc_file_defaults()  
    sns.reset_orig()  
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Analyze each categorical column
for col in cat_cols:
    cat_analyser(df, col)


# In[22]:


df.value_counts()


# In[25]:


df.duplicated().sum()


# In[26]:


df.corr()


# In[27]:


import seaborn as sns

sns.heatmap(df.corr())


# In[28]:


df['depression'].value_counts()


# In[29]:


X = df.drop('depression',axis=1)
X


# In[30]:


y = df['depression']


# In[31]:


print(df['anxiety_level'].mean())
print(df['anxiety_level'].mode())
print(df['anxiety_level'].median())


# In[32]:


(df['anxiety_level'] == 0).sum()


# In[34]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=20)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[35]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[36]:


sns.boxenplot(X_train_scaled)


# In[ ]:




