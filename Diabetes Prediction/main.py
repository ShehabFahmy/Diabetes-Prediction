import pickle
import numpy as np
import scipy as sc
import pandas as pd
import sklearn as sk

df = pd.read_csv("C:/Users/sheha/Downloads/diabetes_binary_health_indicators_BRFSS2015.csv")
dataset_X = df.iloc[:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 19]].values #4 7 9 10 14 16 19
dataset_Y = df.iloc[:,0].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset_X[:], dataset_Y[:], test_size=0.3, random_state=42)

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(x_train, y_train)
x_train = pd.DataFrame(X_resampled)
y_train = pd.DataFrame(y_resampled)

infile = open('svc_model.pkl','rb')
svc_model = pickle.load(open('svc_model.pkl','rb'))
#print(svc_model.score(x_test, y_test))
svc_y_pred = svc_model.predict(x_test)
#print(svc_model.predict([[1, 0, 1, 31, 0, 0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 13]]))
from sklearn import metrics
print("SVM Accuracy: ", metrics.accuracy_score(y_test, svc_y_pred))
print("SVM Precision:", metrics.precision_score(y_test, svc_y_pred))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
print("SVM Confusion Matrix: ", confusion_matrix(y_test, svc_y_pred))
print("SVM F1 Score: ", f1_score(y_test, svc_y_pred))
infile.close()

infile = open('regr_model.pkl','rb')
regr_model = pickle.load(open('regr_model.pkl','rb'))
#print(regr_model.score(x_test, y_test))
regr_y_pred = regr_model.predict(x_test)
#print(regr_model.predict([[1, 0, 1, 31, 0, 0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 13]]))
print("Regression Accuracy: ", metrics.accuracy_score(y_test, regr_y_pred))
print("Regression Precision:", metrics.precision_score(y_test, regr_y_pred))
print("Regression Confusion Matrix: ", confusion_matrix(y_test, regr_y_pred))
print("Regression F1 Score: ", f1_score(y_test, regr_y_pred))
infile.close()

infile = open('id3_model.pkl','rb')
id3_model = pickle.load(open('id3_model.pkl','rb'))
#print(id3_model.score(x_test, y_test))
id3_y_pred = id3_model.predict(x_test)
#print(id3_model.predict([[1, 0, 1, 31, 0, 0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 13]]))
print("ID3 Accuracy: ", metrics.accuracy_score(y_test, id3_y_pred))
print("ID3 Precision:", metrics.precision_score(y_test, id3_y_pred))
print("ID3 Confusion Matrix: ", confusion_matrix(y_test, id3_y_pred))
print("ID3 F1 Score: ", f1_score(y_test, id3_y_pred))
infile.close()

print("Hello!")

cont = True
while cont:
# if not cont:
#     break
    highBP = int(input("Do you have high blood pressure? please enter 1 for yes or 0 for no"))
    if highBP != 1 and highBP != 0:
        print("Invalid Character!")
        cont = False
        break
    highChol = int(input("Do you have high cholesterol? please enter 1 for yes or 0 for no"))
    if highChol != 1 and highChol != 0:
        print("Invalid Character!")
        cont = False
        break
    cholCheck = int(input("Did you check your cholesterol? please enter 1 for yes or 0 for no"))
    if cholCheck != 1 and cholCheck != 0:
        print("Invalid Character!")
        cont = False
        break
    bmi = int(input("Please enter your BMI. please enter a number between 12 and 60"))
    if bmi < 12 or bmi > 60:
        print("Invalid Number!")
        cont = False
        break
    smoker = int(input("Are you a smoker? please enter 1 for yes or 0 for no"))
    if smoker != 1 and smoker != 0:
        print("Invalid Character!")
        cont = False
        break
    stroke = int(input("Do you suffer from any stroke? please enter 1 for yes or 0 for no"))
    if stroke != 1 and stroke != 0:
        print("Invalid Character!")
        cont = False
        break
    heartDisease = int(input("Do you suffer from any heart disease? please enter 1 for yes or 0 for no"))
    if heartDisease != 1 and heartDisease != 0:
        print("Invalid Character!")
        cont = False
        break
    physActivity = int(input("Do you do any physical activity? please enter 1 for yes or 0 for no"))
    if physActivity != 1 and physActivity != 0:
        print("Invalid Character!")
        cont = False
        break
    fruits = int(input("Do you eat fruits? please enter 1 for yes or 0 for no"))
    if fruits != 1 and fruits != 0:
        print("Invalid Character!")
        cont = False
        break
    veggies = int(input("Do you eat vegetables? please enter 1 for yes or 0 for no"))
    if veggies != 1 and veggies != 0:
        print("Invalid Character!")
        cont = False
        break
    hvyAlcoholConsump = int(input("Do you consume alcohols? please enter 1 for yes or 0 for no"))
    if hvyAlcoholConsump != 1 and hvyAlcoholConsump != 0:
        print("Invalid Character!")
        cont = False
        break
    anyHealthCare = int(input("Do you check your health constantly? please enter 1 for yes or 0 for no"))
    if anyHealthCare != 1 and anyHealthCare != 0:
        print("Invalid Character!")
        cont = False
        break
    genHlth = int(input("Rate your general health from 1 to 5"))
    if genHlth < 1 or genHlth > 5:
        print("Invalid Number!")
        cont = False
        break
    mentHlth = int(input("Rate your mental health from 0 to 30"))
    if mentHlth < 0 or mentHlth > 30:
        print("Invalid Number!")
        cont = False
        break
    # diffWalk = int(input("Do you suffer any difficulty in walking? please enter 1 for yes or 0 for no"))
    # if diffWalk != 1 and diffWalk != 0:
    #     print("Invalid Character!")
    #     cont = False
    #     break
    sex = int(input("What is your gender? please enter 0 for female or 1 for male"))
    if sex != 1 and sex != 0:
        print("Invalid Character!")
        cont = False
        break
    age = int(input("Enter your age. please enter a number bigger than 1"))
    if age < 1 or age > 200:
        print("Invalid Number!")
        cont = False
        break
    check = [[highBP, highChol, cholCheck, bmi, smoker, stroke, heartDisease, physActivity, fruits, veggies, hvyAlcoholConsump, anyHealthCare, genHlth, mentHlth, sex, age]]
    m1 = svc_model.predict(check)
    m2 = regr_model.predict(check)
    m3 = id3_model.predict(check)
    if m1 or m2 or m3:
        print("Unfortunately, you are Diabetic")
    else:
        print("Congratulations, you aren't Diabetic")