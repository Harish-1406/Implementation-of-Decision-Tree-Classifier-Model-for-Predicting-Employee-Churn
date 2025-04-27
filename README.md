# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas module and import the required data set.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.
6.Assign the train dataset and test dataset.
7.From sklearn.tree import DecisionTreeClassifier.
8.Use criteria as entropy.
9.From sklearn import metrics.
10.Find the accuracy of our model and predict the require values.
   
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HARISH P K 
RegisterNumber:  212224040104

import pandas as pd
data=pd.read_csv("Employee (1).csv")
data.head()
data.info()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company",
"Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
x.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![image](https://github.com/user-attachments/assets/e7681bf1-bc9d-4872-a62a-b5d3f4ea1ed1)
![image](https://github.com/user-attachments/assets/8adf1c22-cf43-497b-b927-021bea7c327e)
![image](https://github.com/user-attachments/assets/26360f50-ba7d-4f96-b59e-e7082e9a15f8)
![image](https://github.com/user-attachments/assets/ff25958c-6da6-4e90-ba7d-cb370550e196)
![image](https://github.com/user-attachments/assets/e7c89b02-bc6e-4f1a-b038-c2d0478652a1)
![image](https://github.com/user-attachments/assets/3001a973-86b9-4c0a-ae32-e574e5ca642d)






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
