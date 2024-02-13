import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('/Users/ishanchaturvedi/Documents/ML/NET-GUARDIAN/UNSW_NB15_training-set.csv')
test = pd.read_csv('/Users/ishanchaturvedi/Documents/ML/NET-GUARDIAN/UNSW_NB15_testing-set.csv')

object_cols = train.select_dtypes(include = 'object').columns

train.drop(columns = object_cols, inplace = True)
test.drop(columns = object_cols, inplace = True)
df = pd.concat([train, test])

x = df.drop('label', axis = 1)
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

Log_Reg = LogisticRegression(max_iter = 200)
Dec_Tree = DecisionTreeClassifier()
Random_Forest = RandomForestClassifier()
Naive_Bayes = GaussianNB()

Log_Reg.fit(x_train, y_train)
print('Logistic Regression Finished ...')
Dec_Tree.fit(x_train, y_train)
print('Decision Tree Finished ...')
Random_Forest.fit(x_train, y_train)
print('Random Forest Finished ...')
Naive_Bayes.fit(x_train, y_train)
print('Naive Bayes Finished ...')

print(f"Log Reg Accuracy: {Log_Reg.score(x_test, y_test) * 100:.2f}%")
print(f"Dec Tree Accuracy: {Dec_Tree.score(x_test, y_test) * 100:.2f}%")
print(f"Random Forest Accuracy: {Random_Forest.score(x_test, y_test) * 100:.2f}%")
print(f"Naive Bayes Accuracy: {Naive_Bayes.score(x_test, y_test) * 100:.2f}%")

joblib.dump(Log_Reg, 'Log_Reg.joblib')
joblib.dump(Dec_Tree, 'Dec_Tree.joblib')
joblib.dump(Random_Forest, 'Random_Forest.joblib')
joblib.dump(Naive_Bayes, 'Naive_Bayes.joblib')