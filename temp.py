from sklearn.preprocessing import StandardScaler
from pandas.core.common import flatten


values_repository = {'Male' : 1,  'Female' : 0, 'Asymptomatic(ASY)': 0, 'Atypical Angina(ATA)' : 1,
                    'Non-Anginal Pain(NAP)' : 2, 'Typical Angina(TA)' : 3, 'less than 120' : 0,
                    '120 or more' : 1, 'LVH' : 0, 'Normal' : 1, 'ST' :2, 'Yes' : 1, 'No': 0,
                    'Down' : 0, 'Flat' : 1, 'Up' : 2}
patient_details_list = ['souvik', '7797254649', '12', 'Male', 'Asymptomatic(ASY)', '12', '23', 'less than 120', 'LVH', '132', 'Yes', 'Down']
# d is the details list we will get fron the user    '2023-08-21 21:14:58.118607', 



def input_formatter(patient_details_list, values_repository):
    parameter_dict_details = [int(values_repository[element]) if element in values_repository.keys() else int(element) for element in patient_details_list[2:]]
    print(f'parameter details: {parameter_dict_details}\n')
    ss = StandardScaler()
    final_input_list = list()
    list_for_CSV = patient_details_list[0:2]
    list_for_CSV.extend(parameter_dict_details)
    print(list_for_CSV)

    for i in range(len(list_for_CSV[2:])):
        if i in [0, 3, 4, 7]:
            final_input_list.append(ss.fit_transform([[parameter_dict_details[i]]]))
        else:
            final_input_list.append(parameter_dict_details[i])

    final_input_list = list(flatten(final_input_list))
    print(final_input_list)
    return final_input_list, list_for_CSV
    
    
    
input_formatter(patient_details_list, values_repository)