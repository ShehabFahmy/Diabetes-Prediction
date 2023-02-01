import numpy as np
import scipy as sc
import pandas as pd
import sklearn as sk
import pickle

df = pd.read_csv("C:/Users/sheha/Downloads/diabetes_binary_health_indicators_BRFSS2015.csv")
dataset_X = df.iloc[:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 19]].values #4 7 9 10 14 16 19
dataset_Y = df.iloc[:,0].values
#df = df[["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age"]]

# from featurewiz import featurewiz
# features, train = featurewiz(df, 'Diabetes_binary', corr_limit=0.7, verbose=2, sep=",", header=0, test_data="", feature_engg="", category_encoders="")

from collections import Counter
print(sorted(Counter(df['Diabetes_binary']).items()))

# for x in df.index:
#     if df.loc[x, "PhysHlth"] == 0:
#         print(x)
#         df.drop(x, inplace=True)
#
# for y in df.index:
#     if df.loc[y, "BMI"] < 12:
#         print(y)
#         df.drop(y, inplace=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset_X[:], dataset_Y[:], test_size=0.3, random_state=42)

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(x_train, y_train)
print(sorted(Counter(y_resampled).items()))
x_train = pd.DataFrame(X_resampled)
y_train = pd.DataFrame(y_resampled)

from sklearn.svm import SVC
svc = SVC(kernel='linear', random_state = 42)
svc.fit(x_train, y_train)
print(svc.score(x_test, y_test))
y_pred = svc.predict(x_test)
#[1, 1, 1, 30, 1, 0, 0, 0, 1, 5, 30, 1, 0, 9]
#print(y_pred)
from sklearn import metrics
print("SVM Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("SVM Precision:", metrics.precision_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
print("SVM Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("SVM F1 Score: ", f1_score(y_test, y_pred))
print("--------")

# from sklearn.preprocessing import MinMaxScaler
# minmax = MinMaxScaler(feature_range = (0,1))
# data_scaled = minmax.fit_transform(x_train)
# x_train = pd.DataFrame(data_scaled)
# # print(x_train)

from sklearn import linear_model
regr = linear_model.LogisticRegression()
regr.fit(x_train, y_train)
print(regr.score(x_test, y_test))
y_pred = regr.predict(x_test)
#print(y_pred)
print("Regression Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Regression Precision:", metrics.precision_score(y_test, y_pred))
print("Regression Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("Regression F1 Score: ", f1_score(y_test, y_pred))
print("--------")

from sklearn.tree import DecisionTreeClassifier
id3 = DecisionTreeClassifier()
id3.fit(x_train,y_train)
print(id3.score(x_test, y_test))
y_pred = id3.predict(x_test)
#print(y_pred)
print("ID3 Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("ID3 Precision:", metrics.precision_score(y_test, y_pred))
print("ID3 Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("ID3 F1 Score: ", f1_score(y_test, y_pred))
print("--------")

outfile = open('svc_model.pkl','wb')
pickle.dump(svc, outfile)
outfile.close()

outfile = open('regr_model.pkl','wb')
pickle.dump(regr, outfile)
outfile.close()

outfile = open('id3_model.pkl','wb')
pickle.dump(id3, outfile)
outfile.close()

#print(df.isnull().sum())

#print(df.duplicated())