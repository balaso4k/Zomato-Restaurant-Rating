import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Zomato_df.csv')

df.drop('Unnamed: 0', axis=1, inplace=True )
x = df.drop('rate', axis=1)
y = df['rate']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=10)

# Using Exta Tree Regression
from sklearn.ensemble import ExtraTreesRegressor
etr_model = ExtraTreesRegressor(n_estimators=120)
etr_model.fit(x_train,y_train)

y_predict = etr_model.predict(x_test)

import pickle
# Saving model
pickle.dump(etr_model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl','rb'))
print(y_predict)