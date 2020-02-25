import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda

from sklearn.cross_validation import train_test_split

wbcd = pd.read_csv("E:\\Bokey\\Excelr Data\\R Codes\\knn\\wbcd.csv")

wbcd.head(3)
wbcd.drop(["id"],axis=1,inplace=True) # Dropping the uncessary column 
wbcd.columns
wbcd.shape
wbcd.isnull().sum() # No missing values 

#  Malignant as 0 and Beningn as 1

wbcd.loc[wbcd.diagnosis=="B","diagnosis"] = 1
wbcd.loc[wbcd.diagnosis=="M","diagnosis"] = 0

wbcd.diagnosis.value_counts().plot(kind="bar")

train,test = train_test_split(wbcd,test_size = 0.3,random_state=42)
trainX = train.drop(["diagnosis"],axis=1)
trainY = train["diagnosis"]
testX = test.drop(["diagnosis"],axis=1)
testY = test["diagnosis"]

def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1],kernel_initializer="normal",activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer = "rmsprop",metrics = ["accuracy"])
    return model    


first_model = prep_model([30,50,40,20,1])
first_model.fit(np.array(trainX),np.array(trainY),epochs=500)
pred_train = first_model.predict(np.array(trainX))

pred_train = pd.Series([i[0] for i in pred_train])
disease_class = ["B","M"]
pred_train_class = pd.Series(["M"]*398)
pred_train_class[[i>0.5 for i in pred_train]] = "B"

from sklearn.metrics import confusion_matrix
train["original_class"] = "M"
train.loc[train.diagnosis==1,"original_class"] = "B"
train.original_class.value_counts()
confusion_matrix(pred_train_class,train.original_class)
np.mean(pred_train_class==pd.Series(train.original_class).reset_index(drop=True))
pd.crosstab(pred_train_class,pd.Series(train.original_class).reset_index(drop=True))


pred_test = first_model.predict(np.array(testX))
pred_test = pd.Series([i[0] for i in pred_test])
pred_test_class = pd.Series(["M"]*171)
pred_test_class[[i>0.5 for i in pred_test]] = "B"
test["original_class"] = "M"
test.loc[test.diagnosis==1,"original_class"] = "B"
test.original_class.value_counts()
np.mean(pred_test_class==pd.Series(test.original_class).reset_index(drop=True)) # 97.66
pd.crosstab(pred_test_class,test.original_class)
pd.crosstab(test.original_class,pred_test_class).plot(kind="bar")

from keras.utils import plot_model
plot_model(first_model,to_file="first_model.png")

########################## Neural Network for predicting continuous values ###############################

# Reading data 
Concrete = pd.read_csv("E:\\Bokey\\Excelr Data\\Python Codes\\all_py\\Neural Networks\\concrete.csv")
Concrete.head()

def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

column_names = list(Concrete.columns)
predictors = column_names[0:8]
target = column_names[8]

first_model = prep_model([8,50,1])
first_model.fit(np.array(Concrete[predictors]),np.array(Concrete[target]),epochs=900)
pred_train = first_model.predict(np.array(Concrete[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-Concrete[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,Concrete[target],"bo")
np.corrcoef(pred_train,Concrete[target]) # we got high correlation 
