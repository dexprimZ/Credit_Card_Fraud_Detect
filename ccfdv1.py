import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import joblib

data = pd.read_csv('creditcard.csv')

data.shape

print('Rows:', data.shape[0])
print('Columns:', data.shape[1])

data.isnull().sum()

"""Feature Scaling"""

sc = StandardScaler()
data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))

data = data.drop(['Time'], axis=1)

data.duplicated().any()

data = data.drop_duplicates()


"""Distribution Check"""

data['Class'].value_counts()

sns.countplot(data['Class'])

X = data.drop('Class',axis=1)
y = data['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,
                                                 random_state=42)

normal = data[data['Class']==0]
fraud = data[data['Class']==1]

normal_sample=normal.sample(n=473)

new_data = pd.concat([normal_sample,fraud],ignore_index=True)

new_data['Class'].value_counts()

new_data.head()

X = new_data.drop('Class',axis=1)
y = new_data['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,
                                                 random_state=42)

log = LogisticRegression()
log.fit(X_train,y_train)

y_pred1 = log.predict(X_test)


accuracy_score(y_test,y_pred1)

precision_score(y_test,y_pred1)

recall_score(y_test,y_pred1)

f1_score(y_test,y_pred1)

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

y_pred2 = dt.predict(X_test)

accuracy_score(y_test,y_pred2)

precision_score(y_test,y_pred2)

recall_score(y_test,y_pred2)

f1_score(y_test,y_pred2)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred3 = rf.predict(X_test)

accuracy_score(y_test,y_pred3)

precision_score(y_test,y_pred3)

recall_score(y_test,y_pred3)

f1_score(y_test,y_pred3)

final_data = pd.DataFrame({'Models':['LR','DT','RF'],
              "ACC":[accuracy_score(y_test,y_pred1)*100,
                     accuracy_score(y_test,y_pred2)*100,
                     accuracy_score(y_test,y_pred3)*100
                    ]})

final_data

sns.barplot(final_data['Models'], final_data['ACC'])

X = data.drop('Class',axis=1)
y = data['Class']


X_res,y_res = SMOTE().fit_resample(X,y)

y_res.value_counts()

X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.20,
                                                 random_state=42)

log = LogisticRegression()
log.fit(X_train,y_train)

y_pred1 = log.predict(X_test)

accuracy_score(y_test,y_pred1)

precision_score(y_test,y_pred1)

recall_score(y_test,y_pred1)

f1_score(y_test,y_pred1)

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

y_pred2 = dt.predict(X_test)

accuracy_score(y_test,y_pred2)

precision_score(y_test,y_pred2)

recall_score(y_test,y_pred2)

f1_score(y_test,y_pred2)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred3 = rf.predict(X_test)

accuracy_score(y_test,y_pred3)

precision_score(y_test,y_pred3)

recall_score(y_test,y_pred3)

f1_score(y_test,y_pred3)

final_data = pd.DataFrame({'Models':['LR','DT','RF'],
              "ACC":[accuracy_score(y_test,y_pred1)*100,
                     accuracy_score(y_test,y_pred2)*100,
                     accuracy_score(y_test,y_pred3)*100
                    ]})

final_data

sns.barplot(final_data['Models'],final_data['ACC'])

"""Save the model"""

rf1 = RandomForestClassifier()
rf1.fit(X_res,y_res)

import joblib

joblib.dump(rf1,"credit_card_model")

model = joblib.load("credit_card_model")

pred = model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

if pred == 0:
    print("Legit Transcation")
else:
    print("Fraud Transcation")