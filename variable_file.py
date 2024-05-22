import pickle

values_repository = {'Male' : 1,  'Female' : 0, 'Asymptomatic(ASY)': 0, 'Atypical Angina(ATA)' : 1,
                    'Non-Anginal Pain(NAP)' : 2, 'Typical Angina(TA)' : 3, 'less than 120' : 0,
                    '120 or more' : 1, 'LVH' : 0, 'Normal' : 1, 'ST' :2, 'Yes' : 1, 'No': 0,
                    'Down' : 0, 'Flat' : 1, 'Up' : 2}

sex_list = ['Male', 'Female']

chestPainTypeList = ['Asymptomatic(ASY)', 'Atypical Angina(ATA)', 'Non-Anginal Pain(NAP)', 'Typical Angina(TA)']

fastinBS_list = ['less than 120', '120 or more']

restingECG_list = ['LVH', 'Normal', 'ST']

exerciseAngina_list = ['Yes', 'No']

st_Slope_list = ['Down', 'Flat', 'Up']

classifier_output_list = []

classifier_catboost = pickle.load(open('C:/Users/Souvik/VS_Code/trained_categorial_boosting_classifier.sav', 'rb'))
classifier_adaboost = pickle.load(open('C:/Users/Souvik/VS_Code/trained_adaboost_classifier.sav', 'rb'))
classifier_d_tree = pickle.load(open('C:/Users/Souvik/VS_Code/trained_decission_tree.sav', 'rb'))
classifier_gnb = pickle.load(open('C:/Users/Souvik/VS_Code/trained_gaussian_naive_bayes.sav', 'rb'))
classifier_gradient_boosting = pickle.load(open('C:/Users/Souvik/VS_Code/trained_gradient_boosting_classifier.sav', 'rb'))
classifier_knn = pickle.load(open('C:/Users/Souvik/VS_Code/trained_k_nearest_neighbour.sav', 'rb'))
classifier_lightGBM = pickle.load(open('C:/Users/Souvik/VS_Code/trained_lightgbm_classifier.sav', 'rb'))
classifier_LDA = pickle.load(open('C:/Users/Souvik/VS_Code/trained_linear_discriminant_analysis.sav', 'rb'))
classifier_logistic_regression = pickle.load(open('C:/Users/Souvik/VS_Code/trained_logistic_regression.sav', 'rb'))
classifier_multilayer_perceptron = pickle.load(open('C:/Users/Souvik/VS_Code/trained_multilayer_perceptron.sav', 'rb'))
classifier_QDA = pickle.load(open('C:/Users/Souvik/VS_Code/trained_quadratic_discriminant_analysis.sav', 'rb'))
classifier_random_forest = pickle.load(open('C:/Users/Souvik/VS_Code/trained_random_forest.sav', 'rb'))
classifier_SVM = pickle.load(open('C:/Users/Souvik/VS_Code/trained_support_vector_machine.sav', 'rb'))
classifier_xgboost = pickle.load(open('C:/Users/Souvik/VS_Code/trained_xgboost_classifier.sav', 'rb'))

classifier_output_list.append(classifier_catboost)
classifier_output_list.append(classifier_adaboost)
classifier_output_list.append(classifier_d_tree)
classifier_output_list.append(classifier_gnb)
classifier_output_list.append(classifier_gradient_boosting)
classifier_output_list.append(classifier_knn)
classifier_output_list.append(classifier_lightGBM)
classifier_output_list.append(classifier_LDA)
classifier_output_list.append(classifier_logistic_regression)
classifier_output_list.append(classifier_multilayer_perceptron)
classifier_output_list.append(classifier_QDA)
classifier_output_list.append(classifier_random_forest)
classifier_output_list.append(classifier_SVM)
classifier_output_list.append(classifier_xgboost)

# print(classifier_output_list)

output_list = list()
def all_classifiers_op(list_of_classifiers, final_input_list):
    for element in list_of_classifiers:
        output_list.append(element.predict([final_input_list]))
    # print(output_list)
    # for i in output_list:
    #     print(i)
    return 1 if sum(output_list) > len(output_list) // 2 else  0
    # return output_list
    


# all_classifiers_op(classifier_output_list, [0.0, 0, 0, 0.0, 0.0, 0, 1, 0.0, 0, 2])


def valid_phone_no():
    
    return 

def valid_name():
    return 











