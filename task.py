import pandas

#importing the dataset
data = pandas.read_csv("marks1.csv")
print(data)

#Giving feature and target values
feature = data["hrs"]
target  = data["marks"]

#changing the feature to numpy and further to a 2D array
feature = feature.values.reshape(7,1)

#importing LinearRegression form sklearn module
from sklearn.linear_model import LinearRegression

#initializing an empty model
model  = LinearRegression()

#fitting the feature and target values to train the model
model.fit(feature,target)

#predicting the marks for any random value
x = model.predict([[5]])
print(x)

#saving this model for further use

import joblib
joblib.dump(model,"marks.pk1")

#For further use , just we need to load this saved model 
#import joblib
#model1 = joblib.load("marks.pk1")
#model1.predict([[6.5]])


