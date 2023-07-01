# Gerekli Kütüphane ve Fonksiyonlar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.simplefilter(action="ignore")
import matplotlib.pyplot as plt




pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.read_csv(r"C:\Users\Haydar\Desktop\online_shoppers_intention.csv")

df.head()
df.shape
dff = df.copy()
dff.info()
##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(dff)
dff.dtypes

####### BOOOL TİPİNDEKİ DEĞİŞKENLERİN İNTEGERA ÇEVİRİLMESİ #######

dff["Weekend"] = dff["Weekend"].astype("int64")
dff["Revenue"] = dff["Revenue"].astype("int64")
dff.info()
#True/False değerleri 0 ve 1'e çevirdik


####### VALUE SİLME, BİRLEŞTİRME, İSİMLENDİRME İŞLEMİ ##########

##### SİLME İŞLEMİ

dff.drop("SpecialDay", axis=1, inplace=True)
dff.head(10)

##### BİRLEŞTİRME
def merge_rare_categories(dataframe, col, threshold=0.05, new_category_name='Others'):
    unique_values = dataframe[col].value_counts()
    threshold_value = threshold * len(dataframe)/100
    for value in unique_values.index:
        if unique_values[value] < threshold_value:
            dataframe.loc[dataframe[col] == value, col] = new_category_name
    return dataframe.head(10)

merge_rare_categories(dff,"Browser",5)
merge_rare_categories(dff,"Region",5)
merge_rare_categories(dff,"OperatingSystems",20)

##### İSİMLENDİRME

################################Browser##############################
browser_sozluk = {
    1: 'Firefox',
    2: 'Chrome',
    4: 'Internet Explorer'
}

dff["Browser"] = dff["Browser"].replace(browser_sozluk)
##########################OperatingSystems##########################

OperatingSystems_sozluk = {
    1: 'MacOs',
    2: 'Windows',
    3: 'Linux'
}
dff["OperatingSystems"] = dff["OperatingSystems"].replace(OperatingSystems_sozluk)
################################Region##############################
Region_sozluk = {
    1: 'Usa',
    2: 'Canada',
    3: 'Australia',
    4: 'England',
    6: 'France',
    7: 'Germany'
}
dff["Region"] = dff["Region"].replace(Region_sozluk)

##################################
# Veri Görselleştirme
##################################
#Revenue
dff['Revenue'].value_counts()
# checking the Distribution of customers on Revenue
plt.rcParams['figure.figsize'] = (13, 5)
plt.subplot(1, 2, 1)
sns.countplot(dff['Revenue'], palette = 'pastel')
plt.title('Buy or Not', fontsize = 15)
plt.xlabel('Revenue or not', fontsize = 15)
plt.ylabel('count', fontsize = 15)
plt.show(block=True)

#Weekend
dff['Weekend'].value_counts()
# checking the Distribution of customers on Revenue
plt.rcParams['figure.figsize'] = (13, 5)
plt.subplot(1, 2, 1)
sns.countplot(dff['Weekend'], palette = 'pastel')
plt.title('Puchase on Weekends', fontsize = 15)
plt.xlabel('Weekend or not', fontsize = 15)
plt.ylabel('count', fontsize = 15)
plt.show(block=True)

#distribution of Revenue and Weekend data are hightly imbalanced.

#Operating Systems

dff['OperatingSystems'].value_counts()
# plotting a pie chart for Operating Systems

plt.rcParams['figure.figsize'] = (18, 7)
size = [6601, 2585, 2555, 589]
colors = ['violet', 'magenta', 'pink', 'blue']
labels = "Windows", "MacOs", "Linux", "Others"

plt.subplot(1, 2, 2)
plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%', startangle=90)
plt.title('Different Operating Systems', fontsize = 30)
plt.axis('off')
plt.legend()
plt.show(block=True)

#Top 3 Operating Systems are covered 95% of this dataset.1:Windows 2:Linux 3:Macintosh

#Browsers
dff['Browser'].value_counts()
# Ploting a pie chart for operating systems
plt.rcParams['figure.figsize'] = (18, 7)

size = [7961, 2462, 736, 1171]
colors = ['orange', 'yellow', 'pink', 'crimson']
labels = "Chrome", "Firefox", "Internet Explorer", "Others"

plt.subplot(1, 2, 2)
plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.1f%%', startangle = 90)
plt.title('Different Browsers', fontsize = 30)
plt.axis('off')
plt.legend()
plt.show(block=True)

#90% users used only top 3 browser.

#Month
dff['Month'].value_counts()
# plotting a pie chart for share of special days
size = [3364, 2998, 1907, 1727, 549, 448, 433, 432, 288, 184]
colors = ['yellow', 'pink', 'lightblue', 'crimson', 'lightgreen', 'orange', 'cyan', 'magenta', 'violet', 'pink', 'lightblue', 'red']
labels = "May", "November", "March", "December", "October", "September", "August", "July", "June", "February"
explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

circle = plt.Circle((0, 0), 0.6, color = 'white')

plt.rcParams['figure.figsize'] = (18, 7)
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Month', fontsize = 30)
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()
plt.show(block=True)


#Visitor Type
dff['VisitorType'].value_counts()
# plotting a pie chart for Visitors

plt.rcParams['figure.figsize'] = (18, 7)
size = [10551, 1694, 85]
colors = ['lightGreen', 'green', 'pink']
labels = "Returning Visitor", "New Visitor", "Others"
explode = [0, 0, 0.1]
plt.subplot(1, 2, 1)
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Different Visitors', fontsize = 30)
plt.axis('off')
plt.legend()
plt.show(block=True)

#More than 85% visitors are returning vistors

#Traffic Type

dff['TrafficType'].value_counts()
# visualizing the distribution of different traffic around the TrafficType
plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(1, 2, 1)
plt.hist(dff['TrafficType'], color = 'lightblue')
plt.title('Distribution of different Traffic', fontsize = 30)
plt.xlabel('TrafficType Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.grid()
plt.show(block=True)

#Different type of Traffic are not normal(Gaussian) distributed. This data is exponentially distributed. So we need to take care of this type distribution.
#There are 20 different Traffic Type Codes here.

#Region
dff['Region'].value_counts()
# visualizing the distribution of the users around the Region
plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(1, 2, 1)
plt.hist(dff['Region'], color = 'lightgreen')
plt.title('Distribution of users(Customers)', fontsize = 30)
plt.xlabel('Region Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)

plt.show(block=True)

#Different type of users with respect to region are not normal(Gaussian) distributed.This Regional data is exponentially distributed. So we need to take care of this type distribution.
#There are 9 different Region Codes here. 1:United States

##################################
# Değişken v Bagımlı Değişkenin Birlikte Görselleştirilmesi
##################################
#Administrative duration vs Revenue

# boxenplot for Administrative duration vs revenue
plt.rcParams['figure.figsize'] = (8, 5)

sns.boxenplot(dff['Administrative_Duration'], dff['Revenue'], palette = 'pastel', orient='h')
plt.title('Admin. duration vs Revenue', fontsize = 30)
plt.xlabel('Admin. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show(block=True)

#We see here Administrative_Duration is exponentially distributed for both purchased(True) or not puchased(False).
#We also see there are so many outliers in not puchased(False) according to Administrative_Duration.


#Informational duration vs Revenue
# boxenplot for Informational duration vs revenue
plt.rcParams['figure.figsize'] = (8, 5)

sns.boxenplot(dff['Informational_Duration'], dff['Revenue'], palette = 'rainbow', orient = 'h')
plt.title('Info. duration vs Revenue', fontsize = 30)
plt.xlabel('Info. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)

plt.show(block=True)

#We see here Informational_Duration is exponentially distributed for both purchased(True) or not puchased(False).
#We also see there are so many outliers in not puchased(False) according to Informational_Duration.

#Product Related Duration vs Revenue
# boxen plot product related duration vs revenue
plt.rcParams['figure.figsize'] = (8, 5)

sns.boxenplot(dff['ProductRelated_Duration'], dff['Revenue'], palette = 'inferno', orient = 'h')
plt.title('Product Related Duration vs Revenue', fontsize = 30)
plt.xlabel('Product Related Duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show(block=True)
#We see here ProductRelatedDuration is exponentially distributed for both purchased(True) or not puchased(False).
#We also see there are so many outliers in not puchased(False) according to ProductRelatedDuration.

#We see here ExitRates is normally(gaussian) distributed for both purchased(True) or not puchased(False).
#We also see there are so many outliers in not puchased(False) according to ExitRates.

#Page Values vs Revenue
# strip plot for page values vs revenue
plt.rcParams['figure.figsize'] = (8, 5)

sns.stripplot(dff['PageValues'], dff['Revenue'], palette = 'spring', orient = 'h')
plt.title('Page Values vs Revenue', fontsize = 30)
plt.xlabel('PageValues', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show(block=True)
#We see here PageValues is exponentially distributed for both purchased(True) or not puchased(False).
#We also see there are so many outliers in puchased(True) according to ExitRates.
#Most important things is here PageValues are highly influenced to purchased(True) a product.

#Bounce Rates vs Revenue
# strip plot for bounce rates vs revenue
plt.rcParams['figure.figsize'] = (8, 5)

sns.stripplot(dff['BounceRates'], dff['Revenue'], palette = 'autumn', orient = 'h')
plt.title('Bounce Rates vs Revenue', fontsize = 30)
plt.xlabel('Bounce Rates', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show(block=True)

#We see here BounceRates is exponentially distributed for both purchased(True) or not puchased(False).
#We also see there are so many outliers in not puchased(False) according to ExitRates.
#BounceRates is highly influenced to buy a product or not.

#Weekend vs Revenue
# bar plot for weekend vs Revenue
dff = pd.crosstab(dff['Weekend'], dff['Revenue'])
dff.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['orange', 'crimson'])
plt.title('Weekend vs Revenue', fontsize = 30)
plt.show(block=True)

#Traffic Type vs Revenue
# bar plot for traffic type vs revenue

df = pd.crosstab(dff['TrafficType'], dff['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['lightblue', 'blue'])
plt.title('Traffic Type as Revenue', fontsize = 30)
plt.show(block=True)


##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(dff, cat_th=20)
cat_cols.remove("Informational")


cat_cols
num_cols
cat_but_car


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(dff, col)

##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(dff, col, plot=True)
##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(dff, "Revenue", col)

##################################
# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(dff, "Revenue", col)

##################################
# KORELASYON
##################################


dff[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(dff[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

#ya da
dff.corrwith(dff["Revenue"]).sort_values(ascending=False)

##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(dff, na_name=True)

df.isnull().sum()
#yok:)

##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(dff, col))
    if check_outlier(dff, col):
        replace_with_thresholds(dff, col)


##################################
# BASE MODEL KURULUMU
##################################

cat_cols = [col for col in cat_cols if col not in ["Revenue"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)
dff.head()
y = dff["Revenue"]
X = dff.drop(["Revenue"], axis=1)
dff.head()
models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]
for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

########## LR ##########
#Accuracy: 0.8805
#Auc: 0.8136
#Recall: 0.3926
#Precision: 0.7114
#F1: 0.5039
########## KNN ##########
#Accuracy: 0.8616
#Auc: 0.7581
#Recall: 0.3072
#Precision: 0.6086
#F1: 0.4071
########## CART ##########
#Accuracy: 0.8468
#Auc: 0.712
#Recall: 0.5167
#Precision: 0.5429
#F1: 0.5131
########## RF ##########
#Accuracy: 0.8891
#Auc: 0.9062
#Recall: 0.4789
#Precision: 0.7389
#F1: 0.5635
########## XGB ##########
#Accuracy: 0.8841
#Auc: 0.9004
#Recall: 0.5229
#Precision: 0.6915
#F1: 0.573
########## LightGBM ##########
#Accuracy: 0.8862
#Auc: 0.9079
#Recall: 0.5261
#Precision: 0.6995
#F1: 0.5774
########## CatBoost ##########
#Accuracy: 0.8861
#Auc: 0.9149
#Recall: 0.5198
#Precision: 0.6983
#F1: 0.5783


##################################
# ÖZELLİK ÇIKARIMI- FEATURE EXTENSION
##################################
dff['TotalPages'] = dff['Administrative'] + dff['Informational'] + dff['ProductRelated']

dff['TotalTime'] = dff['Administrative_Duration'] + dff['Informational_Duration'] + dff['ProductRelated_Duration']

dff['Adm_time_per_page'] = (dff['Administrative_Duration']/dff['Administrative']).fillna(0)

dff['Info_time_per_page'] = (dff['Informational_Duration']/dff['Informational']).fillna(0)

dff['Pr_time_per_page'] = (dff['ProductRelated_Duration']/dff['ProductRelated']).fillna(0)



#feature ext. sonra bi daha label ve one hot encod.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in dff.columns if dff[col].dtypes == "O" and dff[col].nunique() == 2]
binary_cols
for col in binary_cols:
    dff = label_encoder(dff, col)
#label encoderlık bi durum yok.


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(dff, cat_cols, drop_first=True)

dff.head()


##################################
# MODELLEME
##################################

y = dff["Revenue"]
X = dff.drop(["Revenue"], axis=1)

models = [('LR', LogisticRegression(random_state=17)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=17)),
          ('RF', RandomForestClassifier(random_state=17)),
          ('XGB', XGBClassifier(random_state=17)),
          ("LightGBM", LGBMClassifier(random_state=17)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=17))]
for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
"""
########## LR ##########
Accuracy: 0.8796
Auc: 0.7801
Recall: 0.3853
Precision: 0.7054
F1: 0.4971
########## KNN ##########
Accuracy: 0.8536
Auc: 0.7349
Recall: 0.2369
Precision: 0.5669
F1: 0.3332
########## CART ##########
Accuracy: 0.8454
Auc: 0.7141
Recall: 0.524
Precision: 0.5371
F1: 0.5153
########## RF ##########
Accuracy: 0.8878
Auc: 0.9024
Recall: 0.4684
Precision: 0.7316
F1: 0.5538
########## XGB ##########
Accuracy: 0.8833
Auc: 0.8975
Recall: 0.5119
Precision: 0.6908
F1: 0.5665
########## LightGBM ##########
Accuracy: 0.887
Auc: 0.9089
Recall: 0.5271
Precision: 0.7042
F1: 0.5811
########## CatBoost ##########
Accuracy: 0.8886
Auc: 0.9146
Recall: 0.5266
Precision: 0.7114
F1: 0.5874
"""


################################################
# Random Forests
################################################
#RandomizedSearchCV Ile Hiperparametre Optimizasyonu
rf_params = {'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': range(1,11),
    'min_samples_split': range(1,15,5),
    'n_estimators': [100, 200, 300,500,1000]}
rf = RandomForestClassifier()
rf_randomcv_model = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, n_iter=200, cv=5, scoring='accuracy', n_jobs=-1, verbose=2).fit(X,y)
rf_randomcv_model.best_params_
print('rf randomcv model accuracy score = {}'.format(rf_randomcv_model.best_score_))

#{'n_estimators': 300,
 #'min_samples_split': 11,
 #'min_samples_leaf': 1,
 #'max_features': 3,
 #'max_depth': 100,
 #'bootstrap': True}


rf_model = RandomForestClassifier(random_state=17)
rf_params = {"max_depth": [5, 100, None], # Ağacın maksimum derinliği
             "max_features": [3, 5, 7, "auto"], # En iyi bölünmeyi ararken göz önünde bulundurulması gereken özelliklerin sayısı
             "min_samples_split": [2, 5, 8, 15, 20], # Bir node'u bölmek için gereken minimum örnek sayısı
             "n_estimators": [300, 400, 600]} # Ağaç sayısı
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_ # {'max_depth': None, 'max_features': 7, 'min_samples_split': 8, 'n_estimators': 500}
#rf_final = rf_model.set_params(rf_best_grid.best_params_, random_state=17).fit(X, y)
rf_final = RandomForestClassifier(bootstrap=True, max_depth=100, max_features=3, min_samples_split=11,min_samples_leaf=1, n_estimators=300, random_state=17).fit(X, y)
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1","recall","precision"])
cv_results['test_accuracy'].mean()
#0.8834549878345499
cv_results['test_f1'].mean()
#0.4762199361404512
cv_results['test_recall'].mean()
#0.3651143565720584
cv_results['test_precision'].mean()
#0.7866079572452764


rf_model = RandomForestClassifier(random_state=17)
rf_params = {"max_depth": [5, 8, None], # Ağacın maksimum derinliği
             "max_features": [3, 5, 7, "auto"], # En iyi bölünmeyi ararken göz önünde bulundurulması gereken özelliklerin sayısı
             "min_samples_split": [2, 5, 8, 15, 20], # Bir node'u bölmek için gereken minimum örnek sayısı
             "n_estimators": [100, 200, 500]} # Ağaç sayısı
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_ # {'max_depth': None, 'max_features': 7, 'min_samples_split': 8, 'n_estimators': 500}
#rf_final = rf_model.set_params(rf_best_grid.best_params_, random_state=17).fit(X, y)
rf_final = RandomForestClassifier(max_depth=None, max_features=7,min_samples_split=8,n_estimators=500,random_state=17).fit(X, y)
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1","recall","precision"])
cv_results['test_accuracy'].mean()
#0.8899432278994324
cv_results['test_f1'].mean()
#0.5751767217699368
cv_results['test_recall'].mean()
#0.502468999724441
cv_results['test_precision'].mean()
#0.7270104375858464


###hangi degerler secilmeli karar veremedik?

##Feature Importance:

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final,X)
#page values
#exit rates
#productrelated duration
#total time etc...





