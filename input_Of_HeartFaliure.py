from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
# from scikitplot.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from pandas.core.common import flatten
import numpy as np
import random
import seaborn as sns
import pandas as pd

heart = pd.read_csv('C:/Users/Souvik/Downloads/heart_final.csv')
df1 = heart.copy(deep = True)
colours = ['gold', 'pink', 'plum', 'aqua', 'midnightblue', 'darkorchid', 'greenyellow', 'goldenrod', 'indianred', 'tomato', 'slategray', 'gainsboro', 'tan', 'lime']

le = LabelEncoder()
df1 = heart.copy(deep = True)
# Encoding of data
df1['Sex'] = le.fit_transform(df1['Sex'])
df1['ChestPainType'] = le.fit_transform(df1['ChestPainType'])
df1['RestingECG'] = le.fit_transform(df1['RestingECG'])
df1['ExerciseAngina'] = le.fit_transform(df1['ExerciseAngina'])
df1['ST_Slope'] = le.fit_transform(df1['ST_Slope'])

# Data Scaling
mms = MinMaxScaler()
ss = StandardScaler()
df1['Oldpeak'] = mms.fit_transform(df1[['Oldpeak']])
df1['Age'] = ss.fit_transform(df1[['Age']])
df1['RestingBP'] = ss.fit_transform(df1[['RestingBP']])
df1['Cholesterol'] = ss.fit_transform(df1[['Cholesterol']])
df1['MaxHR'] = ss.fit_transform(df1[['MaxHR']])
df1.head(308)

# Testing
features = df1[df1.columns.drop(['HeartDisease','Oldpeak'])].values
target = df1['HeartDisease'].values
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = 2)


def model(classifier):

    classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)

    print("Accuracy : ",'{0:.2%}'.format(accuracy_score(y_test,prediction)))
    print("Cross Validation Score : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("ROC_AUC Score : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
    # roc_curve(x_test,y_test)
    # plt.title('ROC_AUC_Plot')
    # plt.show()


def model_evaluation(classifier):
    # Confusion Matrix
    cm = confusion_matrix(y_test,classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,cmap = random.sample(colours, 2), fmt ='')
    # Classification Report
    print(classification_report(y_test,classifier.predict(x_test)))
    

classifier_lr = LogisticRegression(random_state = 0,C=10,penalty= 'l2')
print(model(classifier_lr))


def take_inputs(user_input_list):
    '''
    This method is created for only to take input from the user.
    This method calls a bunch of other methods to take inputs and 
    appends the returned value in the input list provided by the user
    and returns a flattened numpy arrray.
    '''
    user_input_list.append(age_input())
    user_input_list.append(sex_input())
    user_input_list.append(chestPainType_input())
    user_input_list.append(restingBP_input())
    user_input_list.append(cholesterol_input())
    user_input_list.append(fastingBS_input())
    user_input_list.append(restingECG_input())
    user_input_list.append(maxHR_input())
    user_input_list.append(exerciseAngina_input())
    user_input_list.append(sT_Slope_input())
    return list(flatten(user_input_list))
    
def age_input():
    age = int(input('Enter the age: '))
    return list(ss.fit_transform([[age]]))

def sex_input():
    sex = int(input('Sex(1 for M/0 for F): '))
    return sex

def chestPainType_input():
    chestPainType = int(input('Enter the chestPainType(0 for ASY/1 for ATA/2 for NAP/3 for TA): '))
    return chestPainType

def restingBP_input():
    restingBP = int(input('Enter the restingBP: '))
    return list(ss.fit_transform([[restingBP]]))

def cholesterol_input():
    cholesterol = int(input('Enter the cholesterol: '))
    return list(ss.fit_transform([[cholesterol]]))

def fastingBS_input():
    fastingBS = int(input('Enter the fastingBS(0 if less than 120/ 1 if greater than or equal to 120): '))
    return fastingBS

def restingECG_input():
    restingECG = int(input('Enter the restingECG(0 for LVH/ 1 for Normal/ 2 for ST): '))
    return restingECG

def maxHR_input():
    maxHR = int(input('Enter the maxHR: '))
    return list(ss.fit_transform([[maxHR]]))

def exerciseAngina_input():
    exerciseAngina = int(input('Enter the exerciseAngina(1 for Yes/0 for No): '))
    return exerciseAngina

def sT_Slope_input():
    sT_Slope = int(input('Enter the sT_Slope(0 for Down/1 for Flat/ 2 for Up): '))
    return sT_Slope

def predict_and_generate_output(final_input_list):
    output_data = classifier_lr.predict([final_input_list])
    print(output_data)
    return 'The person has a possibility to have a heart disease.' if output_data == 1 else 'The person does not has a possibility to have a heart disease.'


user_input_list = list()
final_input_list = take_inputs(user_input_list)
print(f'User_inputlist: \n{user_input_list}')
print(f'final list:\n{final_input_list}')

message = predict_and_generate_output(final_input_list)
print(message)



