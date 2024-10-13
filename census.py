import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('2018_Central_Park_Squirrel_Census_-_Squirrel_Data.csv')

print(data.describe())

print(data)

trimmed_data = data.filter(['X','Y','Primary Fur Color','Age']).dropna()

print(trimmed_data)

x = trimmed_data.drop(columns=["Primary Fur Color","Age"]).values

y = trimmed_data[["Primary Fur Color","Age"]].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#creating instance of decision tree classifier
obj = DecisionTreeClassifier()

obj.fit(X_train, y_train)
predictions = obj.predict(X_test)

print("Enter x coordinate and y coordinate")

x_input,y_input=map(float,input().split())

print(obj.predict([[x_input,y_input]])[0])