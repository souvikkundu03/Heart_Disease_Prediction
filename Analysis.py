import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format
from sklearn.preprocessing import LabelEncoder

def CSV_reader_heart():
    #reading the CVS file and printing the head
    heart = pd.read_csv('C:/Users/Souvik/Downloads/heart_final.csv')
    # print(f'HEAD : \n{heart.head()}')
    return heart

def divider(heart):
    #dividing between numericals and categoricals.
    #Here, categorical features are defined if the the attribute has less than 6 unique elements else it is a numerical feature.
    col = list(heart.columns)
    categorical_features = []
    numerical_features = []
    for i in col:
        if len(heart[i].unique()) > 6:
            numerical_features.append(i)
        else:
            categorical_features.append(i)

    print('\nCategorical Features :',categorical_features)
    print('Numerical Features :',numerical_features)
    mypal= ['#FC05FB', '#FEAEFE', '#FCD2FC','#F3FEFA', '#B4FFE4','#3FFEBA']
    return categorical_features, numerical_features, mypal

def deep_copy_maker(heart):
    # Creating a deep copy of the orginal dataset and label encoding the text data of the categorical features
    le = LabelEncoder()
    df1 = heart.copy(deep = True)

    df1['Sex'] = le.fit_transform(df1['Sex'])
    df1['ChestPainType'] = le.fit_transform(df1['ChestPainType'])
    df1['RestingECG'] = le.fit_transform(df1['RestingECG'])
    df1['ExerciseAngina'] = le.fit_transform(df1['ExerciseAngina'])
    df1['ST_Slope'] = le.fit_transform(df1['ST_Slope'])
    return df1

def info_generator(heart):
    #printing the shape and the columns
    print(f'heart shape : {heart.shape}')
    print(f'heart columns : \n{heart.columns}')

    #printing info 
    print(f'heart info : \n{heart.info}')
    #description of the dataset
    print(f' heart description : \n{heart.describe()}')
    
# def heat_map_plotter(heart):
#     #plotting heatmap to check for null values
#     sns.heatmap(heart.isnull(),cmap = 'magma',cbar = False)
#     plt.show()

#     #plotting the actual heatmap
#     plt.figure(figsize=(7, 7))
#     colormap = sns.color_palette("Blues",120)
#     sns.heatmap(heart.corr(), annot=True,cmap = colormap)
#     plt.show()

def pair_plot_plotter(heart):
    #plotting the pairplot of the dtatset
    sns.pairplot(heart,hue="HeartDisease",plot_kws=dict(marker="+", linewidth=1),diag_kws=dict(fill=False))
    plt.show()

def mean_value_plotter(heart):
    # plotting the mean vaues
    yes = heart[heart['HeartDisease'] == 1].describe().T
    no = heart[heart['HeartDisease'] == 0].describe().T
    colors = ['midnightblue','silver']

    fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (5,5))
    plt.subplot(1,2,1)
    sns.heatmap(yes[['mean']],annot = True,cmap = colors,linewidths = 1,linecolor = 'white',cbar = False,fmt = '.2f',)
    plt.title('Heart Disease');

    plt.subplot(1,2,2)
    sns.heatmap(no[['mean']],annot = True,cmap = colors,linewidths = 1,linecolor = 'white',cbar = False,fmt = '.2f')
    plt.title('No Heart Disease');

    fig.tight_layout(pad = 3)
    tit = 'Mean values of all the features for cases of heart diseases and non-heart diseases'
    fig.suptitle(tit,fontsize=18)
    plt.show()

def violin_plot_plotter(heart):
    #plotting the violin plots of the categorial features
    cat_cols = [ heart.columns[i] for i, j in enumerate(heart.dtypes) if j == 'object']
    plt.figure(figsize=(10,8))

    ax = plt.subplot(2,3,1)
    ax.text(0.5, 0.5, "Violin Plots \n For \n Categorical Columns", fontdict={'fontsize': 20, 'fontweight': 'bold', 'color': 'black', 'ha': 'center', 'va': 'center'})
    ax.axis('off')

    for c, i in enumerate(range(2, 7)):
        plt.subplot(2,3,i)
        cat = cat_cols[c]
        sns.violinplot(x = cat , y = 'HeartDisease', data = heart, saturation =0.9)

    plt.tight_layout()
    plt.show()

def pearson_plotter(heart):
    #pearson's corellation
    df_ = heart[numerical_features]
    corr = df_.corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(8, 5), facecolor=None)
    # cmap = sns.color_palette(mypal, as_cmap=True)
    ax.set_title("Numerical features correlation (Pearson's)", fontsize=20, y= 1.05)
    sns.heatmap(corr, mask=mask, cmap='inferno', vmax=1.0, vmin=-1.0, center=0, annot=True, square=True, linewidths=0, cbar_kws={"shrink": 0.75})
    plt.show()

def variable_distributor(heart):
    #heart disease variable distribution
    plt.figure(figsize=(7, 5),facecolor='white')
    total = float(len(heart))
    ax = sns.countplot(x=heart['HeartDisease'], palette=mypal[1::4])
    ax.set_facecolor('#F6F5F4')

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.1f} %'.format((height/total)*100), ha="center",
               bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))

    ax.set_title('HeartDisease variable distribution', fontsize=20, y=1.05)
    sns.despine(right=True)
    sns.despine(offset=5, trim=True)
    plt.show()

def categorial_features_distributer(categorical_features):
    #distribution of categorial features
    fig, ax = plt.subplots(nrows = 3,ncols = 2,figsize = (8,14))
    for i in range(len(categorical_features) - 1):
        plt.subplot(3,2,i+1)
        sns.distplot(df1[categorical_features[i]],kde_kws = {'bw' : 1},color = 'red');
        title = 'Distribution : ' + categorical_features[i]
        plt.title(title)

    plt.figure(figsize = (4.75,4.55))
    sns.distplot(df1[categorical_features[len(categorical_features) - 1]],kde_kws = {'bw' : 1},color = 'yellow')
    title = 'Distribution : ' + categorical_features[len(categorical_features) - 1]
    plt.title(title)
    plt.show()

def numerical_features_distributor(numerical_features):
    #distribution of numerical features
    fig, ax = plt.subplots(nrows = 2,ncols = 2,figsize = (10,9.75))
    for i in range(len(numerical_features) - 1):
        plt.subplot(2,2,i+1)
        sns.distplot(heart[numerical_features[i]],color = 'midnightblue')
        title = 'Distribution : ' + numerical_features[i]
        plt.title(title)
    plt.show()

    plt.figure(figsize = (4.75,4.55))
    sns.distplot(df1[numerical_features[len(numerical_features) - 1]],kde_kws = {'bw' : 1},color = 'midnightblue')
    title = 'Distribution : ' + numerical_features[len(numerical_features) - 1]
    plt.title(title)
    plt.show()

def difference_generator(heart):
    #difference between Male & Female VS. Numerical Data
    figure = plt.figure(figsize= (12,9))
    ax = plt.subplot(2,3,1)
    ax.text(0.5, 0.5, "Sex Column \n  VS. \n Numerical Varibales \n Hue = HeartDisease", fontdict={'fontsize': 20, 'fontweight': 'bold', 'color': 'black', 'ha': 'center', 'va': 'center'})
    ax.axis('off')
    plt.subplot(2,3,2)
    sns.boxplot(x = 'Sex',y = 'RestingBP',hue = 'HeartDisease',  data = heart, palette = 'crest')
    plt.subplot(2,3,3)
    sns.boxplot(x = 'Sex',y = 'Cholesterol',hue = 'HeartDisease',  data = heart, palette = 'crest')
    plt.subplot(2,3,4)
    sns.boxplot(x = 'Sex',y = 'MaxHR',hue = 'HeartDisease',  data = heart, palette = 'crest')
    plt.subplot(2,3,5)
    sns.boxplot(x = 'Sex',y = 'Age',hue = 'HeartDisease',  data = heart, palette = 'crest')
    plt.subplot(2,3,6)
    sns.boxplot(x = 'Sex', y = 'Oldpeak', hue = 'HeartDisease', data = heart, palette = 'crest')
    plt.tight_layout()
    plt.show()

def categorial_features_vs_poss_heart_disease_generator(heart):
    #categorial features vs positive heart disease cases
    sex = heart[heart['HeartDisease'] == 1]['Sex'].value_counts()
    sex = [sex[0] / sum(sex) * 100, sex[1] / sum(sex) * 100]

    cp = heart[heart['HeartDisease'] == 1]['ChestPainType'].value_counts()
    cp = [cp[0] / sum(cp) * 100,cp[1] / sum(cp) * 100,cp[2] / sum(cp) * 100,cp[3] / sum(cp) * 100]

    fbs = heart[heart['HeartDisease'] == 1]['FastingBS'].value_counts()
    fbs = [fbs[0] / sum(fbs) * 100,fbs[1] / sum(fbs) * 100]

    restecg = heart[heart['HeartDisease'] == 1]['RestingECG'].value_counts()
    restecg = [restecg[0] / sum(restecg) * 100,restecg[1] / sum(restecg) * 100,restecg[2] / sum(restecg) * 100]

    exang = heart[heart['HeartDisease'] == 1]['ExerciseAngina'].value_counts()
    exang = [exang[0] / sum(exang) * 100,exang[1] / sum(exang) * 100]

    slope = heart[heart['HeartDisease'] == 1]['ST_Slope'].value_counts()
    slope = [slope[0] / sum(slope) * 100,slope[1] / sum(slope) * 100,slope[2] / sum(slope) * 100]

    ax,fig = plt.subplots(nrows = 4,ncols = 2,figsize = (15,15))

    plt.subplot(3,2,1)
    plt.pie(sex,labels = ['Male','Female'],autopct='%1.1f%%',startangle = 90,explode = (0.1,0),colors = ['red', 'blue'],wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
    plt.title('Sex')

    plt.subplot(3,2,2)
    plt.pie(cp,labels = ['ASY', 'NAP', 'ATA', 'TA'],autopct='%1.1f%%',startangle = 90,explode = (0,0.1,0.1,0.1),wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
    plt.title('ChestPainType')

    plt.subplot(3,2,3)
    plt.pie(fbs,labels = ['FBS < 120 mg/dl','FBS > 120 mg/dl'],autopct='%1.1f%%',startangle = 90,explode = (0.1,0),colors = ['green', 'yellow'],wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
    plt.title('FastingBS')

    plt.subplot(3,2,4)
    plt.pie(restecg,labels = ['Normal','ST','LVH'],autopct='%1.1f%%',startangle = 90,explode = (0,0.1,0.1),wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
    plt.title('RestingECG')

    plt.subplot(3,2,5)
    plt.pie(exang,labels = ['Angina','No Angina'],autopct='%1.1f%%',startangle = 90,explode = (0.1,0),colors = ['gold', 'silver'],wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
    plt.title('ExerciseAngina')

    plt.subplot(3,2,6)
    plt.pie(slope,labels = ['Flat','Up','Down'],autopct='%1.1f%%',startangle = 90,explode = (0,0.1,0.1),wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
    plt.title('ST_Slope')
    plt.show()

def target_variable_visualizer(heart):
    #Target Variable Visualization (HeartDisease) :
    l = list(heart['HeartDisease'].value_counts())
    circle = [l[1] / sum(l) * 100,l[0] / sum(l) * 100]

    fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (20,5))
    plt.subplot(1,2,1)
    plt.pie(circle,labels = ['No Heart Disease','Heart Disease'],autopct='%1.1f%%',startangle = 90,explode = (0.1,0),colors = ['green', 'yellow'],    wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
    plt.title('Heart Disease %');

    plt.subplot(1,2,2)
    ax = sns.countplot(x='HeartDisease',data = heart,palette = ['blue', 'silver'],edgecolor = 'black')
    for rect in ax.patches:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, rect.get_height(), horizontalalignment='center', fontsize = 11)
    ax.set_xticklabels(['No Heart Disease','Heart Disease'])
    plt.title('Cases of Heart Disease')
    plt.show()

def cat_vs_target_var(categorical_features):
    #Categorical Features vs Target Variable (HeartDisease) :
    fig, ax = plt.subplots(nrows = 3,ncols = 2,figsize = (10,15))
    for i in range(len(categorical_features) - 1):
        plt.subplot(3,2,i+1)
        ax = sns.countplot(x=categorical_features[i],data = heart,hue = "HeartDisease",palette = ['lightblue', 'pink'],edgecolor = 'black')
        for rect in ax.patches:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, rect.get_height(), horizontalalignment='center', fontsize = 11)
        title = categorical_features[i] + ' vs HeartDisease'
        plt.legend(['No Heart Disease','Heart Disease'])
        plt.title(title)
    plt.show()

def num_vs_target_var(numerical_features):
    #Numerical Features vs Target Variable (HeartDisease)
    fig, ax = plt.subplots(nrows = 5,ncols = 1,figsize = (15,30))
    fig.set_figwidth(100)
    for i in range(len(numerical_features)):
        plt.subplot(5,1,i+1)
        sns.countplot(x=numerical_features[i],data = heart,hue = "HeartDisease",palette = ['indianred', 'steelblue'], edgecolor = 'black')
        title = numerical_features[i] + ' vs Heart Disease'
        plt.legend(['No Heart Disease','Heart Disease'])
        plt.title(title)
        fig.subplots_adjust(hspace=1)
    plt.show()

def data_counter(heart):
    #data counting
    heart['RestingBP_Group'] = [ int(i / 5) for i in heart['RestingBP']]
    heart['Cholesterol_Group'] = [ int(i / 10) for i in heart['Cholesterol']]
    heart['MaxHR_Group'] = [ int(i / 5) for i in heart['MaxHR']]
    heart['Oldpeak_Group'] = [ int( (i*10) / 5) for i in heart['Oldpeak']]

    fig, ax = plt.subplots(nrows = 4,ncols = 1,figsize = (10,25))
    group_numerical_features = [i + '_Group' for i in numerical_features[1:]]
    for i in range(len(group_numerical_features)):
        plt.subplot(4,1,i+1)
        sns.countplot(x=group_numerical_features[i],hue = 'HeartDisease',data = heart,palette = ['springgreen', 'plum'], edgecolor = 'black')
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = group_numerical_features[i] + ' vs Heart Disease'
        plt.title(title)
        fig.subplots_adjust(hspace=1)
    plt.show()

def num_features_vs_num_features_wrt_heart_disease(numerical_features):
    #numerical features vs numerical fetures with respect to heart disease
    a = 0
    fig,ax = plt.subplots(nrows = 5,ncols = 2,figsize = (15,25))
    for i in range(len(numerical_features)):
        for j in range(len(numerical_features)):
            if i != j and j > i:
                a += 1
                plt.subplot(5,2,a)
                sns.scatterplot(x = numerical_features[i],y = numerical_features[j],data = heart,hue = 'HeartDisease',palette = ['greenyellow','aqua'], edgecolor = 'black')
                plt.legend(['No Heart Disease', 'Heart Disease'])
                title = numerical_features[i] + ' vs ' + numerical_features[j]
                plt.title(title)
                fig.subplots_adjust(hspace=1)
    plt.show()

heart = CSV_reader_heart()
categorical_features, numerical_features, mypal = divider(heart)
df1 = deep_copy_maker(heart)

def function_caller(heart,categorical_features, numerical_features, mypal,df1):
    info_generator(heart)
    # heat_map_plotter(heart)
    pair_plot_plotter(heart)
    mean_value_plotter(heart)
    violin_plot_plotter(heart)
    pearson_plotter(heart)
    variable_distributor(heart)
    categorial_features_distributer(categorical_features)
    numerical_features_distributor(numerical_features)
    difference_generator(heart)
    categorial_features_vs_poss_heart_disease_generator(heart)
    target_variable_visualizer(heart)
    cat_vs_target_var(categorical_features)
    cat_vs_target_var(categorical_features)
    num_vs_target_var(numerical_features)
    data_counter(heart)
    num_features_vs_num_features_wrt_heart_disease(numerical_features)

# function_caller(heart,categorical_features, numerical_features, mypal,df1)