import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# laod the dataset
df = pd.read_csv("data.csv", header = 0)
      
# cleaning up the dataset
df.drop('Unnamed: 32', axis=1, inplace=True);
df.drop('id', axis=1, inplace=True);

df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0});

# splitting the data set into a 75-25 train/test set
train_set, test_set = train_test_split(df, test_size = 0.25);
train_X = train_set.drop('diagnosis', 1);
train_y = train_set['diagnosis'];
test_X = test_set.drop('diagnosis', 1);
test_y = test_set['diagnosis'];
                           
# model type -- using logistic regression since it works well with discrete data
model=LogisticRegression()

# fitting the model
model.fit(train_X, train_y);
         
# Obtaining accuracy for training set
train_predictions = model.predict(train_X);
train_accuracy = metrics.accuracy_score(train_predictions, train_y);
print("Training Set Accuracy: %s" % "{0:.2%}".format(train_accuracy));

# Obtaining accuracy for test set
test_predictions = model.predict(test_X);
test_accuracy = metrics.accuracy_score(test_predictions, test_y);
print("Test Set Accuracy: %s" % "{0:.2%}".format(test_accuracy));