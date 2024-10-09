import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv(r"C:\Users\jayas\Downloads\Customer_churn_project\Telco-Customer-Churn.csv")

df.head()

df.info()

df.describe()

df.isna().sum()

df.shape

df.columns

df.dropna(how = 'any',inplace = True)

bins = [0,12,24,36,48,60,72]
labels = ['1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72']
df['tenure_group'] = pd.cut(df['tenure'],bins = bins,labels = labels,right = False)

df.drop(['customerID','tenure'],inplace = True,axis = 1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

categ=['gender','SeniorCitizen', 'tenure_group' ,'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod',  'Churn',]
df[categ] = df[categ].apply(le.fit_transform)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df['TotalCharges'].astype(int)

x = df.drop('Churn',axis= 1)
y = df['Churn']
x = x.replace(' ',np.nan)
x = x.apply(pd.to_numeric, errors='coerce')
x = x.fillna(x.mean())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=0)

'''without HyperParameters
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier()
model1.fit(x_train,y_train)

model1.score(x_test,y_test) 
acc = 0.72
'''
#with HyperParameter Tuning
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(criterion='gini',random_state=100,max_depth=6,max_leaf_nodes= 6)
model1.fit(x_train,y_train)

model1.score(x_test,y_test)

#acc = 0.78

y_pred_dt = model1.predict(x_test)

from sklearn.metrics import classification_report,roc_auc_score,roc_curve
classification_report(y_test,y_pred_dt,labels = [0,1])

from imblearn.over_sampling import SMOTE
s = SMOTE()
x_ovs,y_ovs = s.fit_resample(x,y)

x1_train,x1_test,y1_train,y1_test = train_test_split(x_ovs,y_ovs,random_state = 0,test_size=0.20)

dt_model = DecisionTreeClassifier(criterion='gini',random_state=0,max_depth=6,max_leaf_nodes=6)
dt_model.fit(x1_train,y1_train)

dt_acc = dt_model.score(x1_test,y1_test)

#LogesticRegression Model
from sklearn.linear_model import LogisticRegression
le = LogisticRegression()
le_model = le.fit(x1_train,y1_train)

le_acc = le_model.score(x1_test,y1_test)

le_pred = le_model.predict(x1_test)

classification_report(le_pred,y1_test,labels = [0,1])
le_probs = le_model.predict_proba(x1_test)[:,1]
fpr1, tpr1, thresholds1 = roc_curve(y1_test, le_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LogisticRegression (ROC Curve)')
plt.legend()
plt.show()

le_auc_score = roc_auc_score(y1_test, le_probs)
print(f"AUC Score: {le_auc_score:.2f}")

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(le_pred,y1_test)
acc1 = accuracy_score(le_pred,y1_test)


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_model = rf.fit(x1_train,y1_train)

rf_acc = rf_model.score(x1_test,y1_test)

rf_pred = rf_model.predict(x1_test)
classification_report(rf_pred,y1_test,labels = [0,1])
confusion_matrix(rf_pred,y1_test)
acc2 = accuracy_score(rf_pred,y1_test)

rf_probs = rf_model.predict_proba(x1_test)[:,1]
fpr2,tpr2,thresholds2 = roc_curve(y1_test,rf_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr2, tpr2, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RandomForestClassifier (ROC Curve)')
plt.legend()
plt.show()

rf_auc_score = roc_auc_score(y1_test,rf_probs)
print(f"AUC Score: {rf_auc_score:.2f}")


#KNN MOdel
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3,algorithm='ball_tree')
knn_model = knn.fit(x1_train,y1_train)

knn_acc = knn_model.score(x1_test,y1_test)
knn_pred = knn_model.predict(x1_test)
classification_report(knn_pred,y1_test,labels = [0,1])
confusion_matrix(knn_pred,y1_test)
acc4 = accuracy_score(knn_pred,y1_test)

knn_probs = knn_model.predict_proba(x1_test)[:,1]
fpr4,tpr4,thresholds4 = roc_curve(y1_test,knn_probs)

knn_acc_score = roc_auc_score(y1_test,knn_probs)
print(f"AUC Score: {knn_acc_score:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr4, tpr4, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN Model (ROC Curve)')
plt.legend()
plt.show()


#Adaboost
from sklearn.ensemble import AdaBoostClassifier
ad = AdaBoostClassifier(n_estimators = 60,random_state= 100 )
ad_model = ad.fit(x1_train,y1_train)
ad_acc = ad_model.score(x1_test,y1_test)
ad_pred = ad_model.predict(x1_test)
classification_report(ad_pred,y1_test,labels = [0,1])
confusion_matrix(ad_pred,y1_test)
acc5 = accuracy_score(ad_pred,y1_test)

ad_probs = ad_model.predict_proba(x1_test)[:,1]
fpr5,tpr5,thresholds5 = roc_curve(y1_test,ad_probs)
ad_acc_score = roc_auc_score(y1_test,ad_probs)
print(f"AUC Score: {ad_acc_score:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr5, tpr5, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Adaboost (ROC Curve)')
plt.legend()
plt.show()


#Xgboost
from xgboost import XGBClassifier
xg = XGBClassifier(class_weight={0:1, 1:2})
xg_model = xg.fit(x1_train,y1_train)
xg_acc = xg_model.score(x1_test,y1_test)
xg_pred = xg_model.predict(x1_test)
classification_report(xg_pred,y1_test,labels = [0,1])
confusion_matrix(xg_pred,y1_test)
acc6 = accuracy_score(xg_pred,y1_test)

xg_probs = xg_model.predict_proba(x1_test)[:,1]
fpr6,tpr6,thresholds6 = roc_curve(y1_test,xg_probs)
xg_acc_score = roc_auc_score(y1_test,xg_probs)
print(f"AUC Score: {xg_acc_score:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr6, tpr6, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Xgboost (ROC Curve)')
plt.legend()
plt.show()

#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators= 100, min_samples_split=5 , max_depth= 7, learning_rate= 0.1)
gb_model = gb.fit(x1_train,y1_train)
gb_acc = gb_model.score(x1_test,y1_test)
gb_pred = gb_model.predict(x1_test)
classification_report(gb_pred,y1_test,labels = [0,1])
confusion_matrix(gb_pred,y1_test)
acc7 = accuracy_score(gb_pred,y1_test)

gb_probs = gb_model.predict_proba(x1_test)[:,1]
fpr7,tpr7,thresholds7 = roc_curve(y1_test,gb_probs)
gb_acc_score = roc_auc_score(y1_test,gb_probs)
print(f"AUC Score: {gb_acc_score:.2f}")


plt.figure(figsize=(8, 6))
plt.plot(fpr7, tpr7, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GradientBoostingClassifier (ROC Curve)')
plt.legend()
plt.show()

with open('BestFitmodel.pkl','wb') as file:
    pickle.dump(gb_model,file)
with open('BestFitmodel.pkl','rb') as file:
    loaded_model = pickle.load(file)
   

