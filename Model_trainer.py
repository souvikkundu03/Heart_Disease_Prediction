import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
from Analysis import df1, categorical_features, numerical_features
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import SelectKBest,chi2,f_classif
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle

#Data Scaling 
mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization

df1['Oldpeak'] = mms.fit_transform(df1[['Oldpeak']])
df1['Age'] = ss.fit_transform(df1[['Age']])
df1['RestingBP'] = ss.fit_transform(df1[['RestingBP']])
df1['Cholesterol'] = ss.fit_transform(df1[['Cholesterol']])
df1['MaxHR'] = ss.fit_transform(df1[['MaxHR']])
print(df1.head(308))

print(numerical_features)
colours = ['gold', 'pink', 'plum', 'aqua', 'midnightblue', 'darkorchid', 'greenyellow', 'goldenrod', 'indianred', 'tomato', 'slategray', 'gainsboro', 'tan', 'lime']

def correlation_matrix_plotter(df1):
    #plotting correlation matrix
    plt.figure(figsize = (20,5))
    sns.heatmap(df1.corr(),cmap = ['aqua','plum'],annot = True)
    plt.show()

def correlation_matrix_wrt_heart_fisease(df1):
    # the correlation only with respect to HeartDiseas
    corr = df1.corrwith(df1['HeartDisease']).sort_values(ascending = False).to_frame()
    corr.columns = ['Correlations']
    plt.subplots(figsize = (5,5))
    sns.heatmap(corr,annot = True,cmap = ['teal', 'tan'],linewidths = 0.4,linecolor = 'black');
    plt.title('Correlation w.r.t HeartDisease')
    plt.show()
    
def feature_selection_from_categorial_features(categorical_features):
    #chi square test
    features = df1.loc[:,categorical_features[:-1]]
    target = df1.loc[:,categorical_features[-1]]

    best_features = SelectKBest(score_func = chi2,k = 'all')
    fit = best_features.fit(features,target)

    featureScores = pd.DataFrame(data = fit.scores_,index = list(features.columns),columns = ['Chi Squared Score'])

    plt.subplots(figsize = (5,5))
    sns.heatmap(featureScores.sort_values(ascending = False,by = 'Chi Squared Score'),annot = True,cmap = ['khaki', 'darkorchid'],linewidths = 0.4,linecolor = 'black',fmt = '.2f');
    plt.title('Selection of Categorical Features')
    plt.show()

def feature_selection_from_numerical_features(numerical_features):
    #ANOVA test
    features = df1.loc[:,numerical_features]
    target = df1.loc[:,categorical_features[-1]]

    best_features = SelectKBest(score_func = f_classif,k = 'all')
    fit = best_features.fit(features,target)

    featureScores = pd.DataFrame(data = fit.scores_,index = list(features.columns),columns = ['ANOVA Score'])

    plt.subplots(figsize = (5,5))
    sns.heatmap(featureScores.sort_values(ascending = False,by = 'ANOVA Score'),annot = True,cmap = ['c', 'lightsteelblue'],linewidths = 0.4,linecolor = 'black',fmt = '.2f');
    plt.title('Selection of Numerical Features')
    plt.show()

def dataset_splitter(df1):
    features = df1[df1.columns.drop(['HeartDisease','Oldpeak'])].values
    target = df1['HeartDisease'].values
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = 42)
    return x_train, x_test, y_train, y_test

def model(classifier):
    #fitting the classifiers
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
    print("Accuracy : ",'{0:.2%}'.format(accuracy_score(y_test,prediction)))
    print("Cross Validation Score : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("ROC_AUC Score : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
    
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
    plt.show()
    
def Logistic_Regression(x_train,y_train):
    classifier_lr = LogisticRegression(random_state = 0,C=10,penalty= 'l2')
    classifier_lr.fit(x_train,y_train)
    model(classifier_lr)
    model_evaluation(classifier_lr)
    return classifier_lr

def Support_vector_Machine(x_train,y_train):
    classifier_svc = SVC(kernel = 'linear',C = 0.1)
    classifier_svc.fit(x_train,y_train)
    model(classifier_svc)
    model_evaluation(classifier_svc)
    return classifier_svc

def Decission_Tree(x_train,y_train):
    classifier_dt = DecisionTreeClassifier(random_state = 1000,max_depth = 4,min_samples_leaf = 1)
    classifier_dt.fit(x_train,y_train)
    model(classifier_dt)
    model_evaluation(classifier_dt)
    return classifier_dt

def Random_Forest(x_train,y_train):
    classifier_rf = RandomForestClassifier(max_depth = 4,random_state = 0)
    classifier_rf.fit(x_train,y_train)
    model(classifier_rf)
    model_evaluation(classifier_rf)
    return classifier_rf

def K_Nearest_Neighbour(x_train,y_train):
    classifier_knn = KNeighborsClassifier(leaf_size = 1, n_neighbors = 3,p = 1)
    classifier_knn.fit(x_train,y_train)
    model(classifier_knn)
    model_evaluation(classifier_knn)
    return classifier_knn
    
def Guassian_Naive_Bayes(x_train,y_train):
    classifier_gnb = GaussianNB()
    classifier_gnb.fit(x_train,y_train)
    model(classifier_gnb)
    model_evaluation(classifier_gnb)
    return classifier_gnb

def LDA(x_train, y_train):
    classifier_lda = LinearDiscriminantAnalysis()
    classifier_lda.fit(x_train,y_train)
    model(classifier_lda)
    model_evaluation(classifier_lda)
    return classifier_lda

def QDA(x_train, y_train):
    classifier_qda = QuadraticDiscriminantAnalysis()
    classifier_qda.fit(x_train,y_train)
    model(classifier_qda)
    model_evaluation(classifier_qda)
    return classifier_qda

def AdaBoost_Classifier(x_train,y_train):
    classifier_adb = AdaBoostClassifier()
    classifier_adb.fit(x_train,y_train)
    model(classifier_adb)
    model_evaluation(classifier_adb)
    return classifier_adb

def Gradient_Boosting_classifier(x_train,y_train):
    classifier_gbc = GradientBoostingClassifier()
    classifier_gbc.fit(x_train,y_train)
    model(classifier_gbc)
    model_evaluation(classifier_gbc)
    return classifier_gbc

def Multilayer_perceptron(x_train,y_train):
    classifier_mlp = MLPClassifier()
    classifier_mlp.fit(x_train,y_train)
    model(classifier_mlp)
    model_evaluation(classifier_mlp)
    return classifier_mlp

def Cat_boost(x_train, y_train):
    classifier_cbc = CatBoostClassifier()
    classifier_cbc.fit(x_train, y_train)
    model(classifier_cbc)
    model_evaluation(classifier_cbc)
    return classifier_cbc

def xg_boost(x_train, y_train):
    classsifier_xgbc = XGBClassifier()
    classsifier_xgbc.fit(x_train, y_train)
    model(classsifier_xgbc)
    model_evaluation(classsifier_xgbc)
    return classsifier_xgbc

def  lightGBM_classifier(x_train,y_train):
    classsifier_lmg = LGBMClassifier()
    classsifier_lmg.fit(x_train,y_train)
    model(classsifier_lmg)
    model_evaluation(classsifier_lmg)
    return classsifier_lmg

def model_saver(x_train,y_train):
    #logistic regression model
    print('LOGISTIC REGRESSION : \n')
    classifier_lr = Logistic_Regression(x_train,y_train)
    pickle.dump(classifier_lr, open('trained_logistic_regression.sav', 'wb'))

    # support vector machine
    print('SUPPORT VECTOR MACHINE:\n')
    classifier_svc = Support_vector_Machine(x_train,y_train)
    pickle.dump(classifier_svc, open('trained_support_vector_machine.sav', 'wb'))

    #decission tree
    print('DECISSION TREE:\n')
    classifier_dt = Decission_Tree(x_train,y_train)
    pickle.dump(classifier_dt, open('trained_decission_tree.sav', 'wb'))
    
    #Random Forest
    print('RANDOM FPREST:\n')
    classifier_rf = Random_Forest(x_train,y_train)
    pickle.dump(classifier_rf, open('trained_random_forest.sav', 'wb'))
    
    #KNN
    print('K NEAREST NEIGHBOUR:\n')
    classifier_knn = K_Nearest_Neighbour(x_train,y_train)
    pickle.dump(classifier_knn, open('trained_k_nearest_neighbour.sav', 'wb'))
    
    #Gausian Naive Bayes
    print('GAUSSIAN NAIVE BAYES:\n')
    classifier_gnb = Guassian_Naive_Bayes(x_train,y_train)
    pickle.dump(classifier_gnb, open('trained_gaussian_naive_bayes.sav', 'wb'))
    
    #LDA
    print('LINEAR DISCRIMINANT ANALYSIS:\n')
    classifier_lda = LDA(x_train,y_train)
    pickle.dump(classifier_lda, open('trained_linear_discriminant_analysis.sav', 'wb'))
    
    #QDA
    print('QUADRATIC DISCRIMINANT ANALYSIS:\n')
    classifier_qda = QDA(x_train,y_train)
    pickle.dump(classifier_qda, open('trained_quadratic_discriminant_analysis.sav', 'wb'))
    
    #AdaBoost classifier
    print('ADABOOST:\n')
    classifier_adb = AdaBoost_Classifier(x_train,y_train)
    pickle.dump(classifier_adb, open('trained_adaboost_classifier.sav', 'wb'))
    
    #Gradient boosting classifier
    print('GRADIENT BOOSTING CLASSIFIER:\n')
    classifier_gbc = Gradient_Boosting_classifier(x_train,y_train)
    pickle.dump(classifier_gbc, open('trained_gradient_boosting_classifier.sav', 'wb'))
    
    #Multilayer Perceptron
    print('MULTI LAYER PERCEPTRON:\n')
    classifier_mlp = Multilayer_perceptron(x_train,y_train)
    pickle.dump(classifier_mlp, open('trained_multilayer_perceptron.sav', 'wb'))
    
    #Cat Boost
    print('CATEGORIAL BOOSTING CLASSIFIER:\n')
    classifier_cbc = Cat_boost(x_train,y_train)
    pickle.dump(classifier_cbc, open('trained_categorial_boosting_classifier.sav', 'wb'))
    
    #XGBoost
    print('XGBOOST CLASSIFIER:\n')
    classsifier_xgbc = xg_boost(x_train,y_train)
    pickle.dump(classsifier_xgbc, open('trained_xgboost_classifier.sav', 'wb'))
    
    #LightGBM
    print('LIGHTGBM CLASSIFIER:\n')
    classsifier_lmg = lightGBM_classifier(x_train,y_train)
    pickle.dump(classsifier_lmg, open('trained_lightgbm_classifier.sav', 'wb'))

def model_testing_score_evaluation(x_test, y_test):
    
    classifier_lr = pickle.load(open('trained_logistic_regression.sav', 'rb'))
    result = str(classifier_lr.score(x_test, y_test) * 100) + "%"
    print(f'LOGISTIC REGRESSION : {result}\n')
    
    classifier_svc = pickle.load(open('trained_support_vector_machine.sav', 'rb'))
    result = str(classifier_svc.score(x_test, y_test) * 100) + "%"
    print(f'SUPPORT VECTOR MACHINE : {result}\n')
    
    classifier_dt = pickle.load(open('trained_decission_tree.sav', 'rb'))
    result = str(classifier_dt.score(x_test, y_test) * 100) + "%"
    print(f'DECISSION TREE : {result}\n')
    
    classifier_rf = pickle.load(open('trained_random_forest.sav', 'rb'))
    result = str(classifier_rf.score(x_test, y_test) * 100) + "%"
    print(f'RANDOM FPREST : {result}\n')
    
    classifier_knn = pickle.load(open('trained_k_nearest_neighbour.sav', 'rb'))
    result = str(classifier_knn.score(x_test, y_test) * 100) + "%"
    print(f'K NEAREST NEIGHBOUR : {result}\n')
    
    classifier_gnb = pickle.load(open('trained_gaussian_naive_bayes.sav', 'rb'))
    result = str(classifier_gnb.score(x_test, y_test) * 100) + "%"
    print(f'GAUSSIAN NAIVE BAYES : {result}\n')
    
    classifier_lda = pickle.load(open('trained_linear_discriminant_analysis.sav', 'rb'))
    result = str(classifier_lda.score(x_test, y_test) * 100) + "%"
    print(f'LINEAR DISCRIMINANT ANALYSIS : {result}\n')
    
    classifier_qda = pickle.load(open('trained_quadratic_discriminant_analysis.sav', 'rb'))
    result = str(classifier_qda.score(x_test, y_test) * 100) + "%"
    print(f'QUADRATIC DISCRIMINANT ANALYSIS : {result}\n')
    
    classifier_adb = pickle.load(open('trained_adaboost_classifier.sav', 'rb'))
    result = str(classifier_adb.score(x_test, y_test) * 100) + "%"
    print(f'ADABOOST : {result}\n')
    
    classifier_gbc = pickle.load(open('trained_gradient_boosting_classifier.sav', 'rb'))
    result = str(classifier_gbc.score(x_test, y_test) * 100) + "%"
    print(f'GRADIENT BOOSTING CLASSIFIER : {result}\n')
    
    classifier_mlp = pickle.load(open('trained_multilayer_perceptron.sav', 'rb'))
    result = str(classifier_mlp.score(x_test, y_test) * 100) + "%"
    print(f'MULTI LAYER PERCEPTRON : {result}\n')
    
    classifier_cbc = pickle.load(open('trained_categorial_boosting_classifier.sav', 'rb'))
    result = str(classifier_cbc.score(x_test, y_test) * 100) + "%"
    print(f'CATEGORIAL BOOSTING CLASSIFIER : {result}\n')
    
    classsifier_xgbc = pickle.load(open('trained_xgboost_classifier.sav', 'rb'))
    result = str(classsifier_xgbc.score(x_test, y_test) * 100) + "%"
    print(f'XGBOOST CLASSIFIER : {result}\n')
    
    classsifier_lmg = pickle.load(open('trained_lightgbm_classifier.sav', 'rb'))
    result = str(classsifier_lmg.score(x_test, y_test) * 100) + "%"
    print(f'LIGHTGBM CLASSIFIER : {result}\n')

# correlation_matrix_plotter(df1)
# correlation_matrix_wrt_heart_fisease(df1)
# feature_selection_from_categorial_features(categorical_features)
# feature_selection_from_numerical_features(numerical_features)
x_train, x_test, y_train, y_test = dataset_splitter(df1)
# model_saver(x_train,y_train)
# model_testing_score_evaluation(x_test, y_test)