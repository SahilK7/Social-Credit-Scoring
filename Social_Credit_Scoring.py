#Social Credit Scoring
# Data Preprocessing 
# Importing the libraries
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset1 = pd.read_csv('cs-training.csv')
dataset2 = pd.read_csv('cs-test.csv')

#Data Sample
sample1 = dataset1.sample(10)
sample2 = dataset2.sample(10)

sa = dataset1.head(20)


dataset1.info()


#large null values for MonthlyIncome and NumberOfDependents, these features has 
#inconsistentdata types, we will change them to int64. 

summary =dataset1.describe()


#Age feature have an outlier value 0. I assume that it is not recorded
#and we will impute with the age's median. 

#NumberOfTimes90DaysLate, NumberOfTime60-89DaysPastDueNotWorse and NumberOfTime30-59DaysPastDueNotWorse 
#looks like giving the same information. 

#Also, NumberOfOpenCreditLinesAndLoans and NumberRealEstateLoansOrLines. 
#We will check their correlations to each other with the correlation matrix.

dataset2.info()
summary1 = dataset2.describe()

# min value of the age feature is 21 in dataset2. 

#Checking the distribution of our target class.
plt.figure(figsize=(10,8))
sns.countplot("SeriousDlqin2yrs", data=dataset1)
# Just to check age outlier with 0 value with box plot
import seaborn as sns
sns.boxplot(x='age',data=dataset1)
#Just to check age outlier with 0 value with scatter plot
plt.figure(figsize=(10,8))
plt.scatter(dataset1['age'],dataset1['MonthlyIncome'])


dataset1.loc[:, 'age'].head(5)
value = dataset1.loc[dataset1["MonthlyIncome"]==3000]

dataset1.get_dtype_counts()
sample = dataset1.head(20)

#There is clear problem here, we have an unbalanced target class!! we will 
#check the event rate of financial distress (SeriousDlqin2yrs) in our dataset.
class_0 = dataset1.SeriousDlqin2yrs.value_counts()[0]
class_1 = dataset1.SeriousDlqin2yrs.value_counts()[1]
print("Total number of class_0: {}".format(class_0))
print("Total number of class_1: {}".format(class_1))
print("Event rate: {} %".format(class_1/(class_0+class_1) *100))


#age feature has a 0 value in it, 
# impute it with the age median.
dataset1.loc[dataset1["age"] < 18] #less than legal age
#only one instance
dataset1.loc[dataset1["age"] == 0, "age"] = dataset1.age.median()

dataset1.iloc[65695]


age_working = dataset1.loc[(dataset1["age"] >= 18) & (dataset1["age"] < 60)]
age_senior = dataset1.loc[(dataset1["age"] >= 60)]

age_working_impute = age_working.MonthlyIncome.mean()
age_senior_impute = age_senior.MonthlyIncome.mean()

#We will change the monthlyincome data type to int64 then fill those null 
#values with 99999 and impute with the corresponding age's monthlyincome mean.
dataset1["MonthlyIncome"] = np.absolute(dataset1["MonthlyIncome"])

dataset1["MonthlyIncome"] = dataset1["MonthlyIncome"].fillna(99999)

dataset1["MonthlyIncome"] = dataset1["MonthlyIncome"].astype('int64')

dataset1.loc[((dataset1["age"] >= 18) & (dataset1["age"] < 60)) & (dataset1["MonthlyIncome"] == 99999),\
               "MonthlyIncome"] = age_working_impute
dataset1.loc[(dataset1["age"] >= 60) & (dataset1["MonthlyIncome"] == 99999), "MonthlyIncome"] = age_senior_impute

#check
dataset1.info()

dataset1.loc[dataset1["MonthlyIncome"] == 99999]


#We're done with the Monthly Income, now we will move to the NumberOfDependents feature.

dataset1["NumberOfDependents"] = np.absolute(dataset1["NumberOfDependents"])
dataset1["NumberOfDependents"] = dataset1["NumberOfDependents"].fillna(0)
dataset1["NumberOfDependents"] = dataset1["NumberOfDependents"].astype('int64')

dataset1.NumberOfDependents.value_counts()

dataset1.info()

#I decided not to go through each of the numberofdependents feature and impute it by the mode.

#We will now take a look at the correlation of the features to the target variable.
corr = dataset1.corr()
#plt.figure(figsize = (18,17))
plt.rcParams.update({'font.size':10})
sns.heatmap(corr, annot = True, fmt = ".2g")
plt.savefig('file.png')

#avoiding multicollinearity
#2 ways to handle this, drop the other 2 features and keep 1 or combine 
#the three features and make a binary feature that classify if a borrower defaulted any loan/credit payment.
#Also, the NumberOfOpenCreditLinesAndLoans and NumberRealEstateLoansOrLines features are somehow correlated 
#to each other but has different degree of correlation from our target class.


dataset1["CombinedDefaulted"] = (dataset1["NumberOfTimes90DaysLate"] + dataset1["NumberOfTime60-89DaysPastDueNotWorse"])\
                                        + dataset1["NumberOfTime30-59DaysPastDueNotWorse"]

dataset1.loc[(dataset1["CombinedDefaulted"] >= 1), "CombinedDefaulted"] = 1

dataset1["CombinedCreditLoans"] = dataset1["NumberOfOpenCreditLinesAndLoans"] + \
                                        dataset1["NumberRealEstateLoansOrLines"]

dataset1.loc[(dataset1["CombinedCreditLoans"] <= 5), "CombinedCreditLoans"] = 0
dataset1.loc[(dataset1["CombinedCreditLoans"] > 5), "CombinedCreditLoans"] = 1

dataset1.CombinedCreditLoans.value_counts()

#Next, we will create a binary feature WithDependents which is derived from the 
#NumberOfDependents feature. Also, from the description of the data DebtRatio = Monthly debt payments / monthly gross income. 
#we will extract MonthlyDebtPayments from this formula to get a new feature.

dataset1["WithDependents"] = dataset1["NumberOfDependents"]
dataset1.loc[(dataset1["WithDependents"] >= 1), "WithDependents"] = 1

dataset1.WithDependents.value_counts()


dataset1["MonthlyDebtPayments"] = dataset1["DebtRatio"] * dataset1["MonthlyIncome"]
dataset1["MonthlyDebtPayments"] = np.absolute(dataset1["MonthlyDebtPayments"])
dataset1["MonthlyDebtPayments"] = dataset1["MonthlyDebtPayments"].astype('int64')


dataset1["age"] = dataset1["age"].astype('int64')
dataset1["MonthlyIncome"] = dataset1["MonthlyIncome"].astype('int64')


#Also, let's see if we can get a good predictor out of age feature. using 
#senior and working temporary dataframes earlier.
dataset1["age_map"] = dataset1["age"]
dataset1.loc[(dataset1["age"] >= 18) & (dataset1["age"] < 60), "age_map"] = 1
dataset1.loc[(dataset1["age"] >= 60), "age_map"] = 0 

#replacing those numbers to categorical features then get the dummy variables
dataset1["age_map"] = dataset1["age_map"].replace(0, "working")
dataset1["age_map"] = dataset1["age_map"].replace(1, "senior")
#Convert categorical variable into dummy/indicator variables.
dataset1 = pd.concat([dataset1, pd.get_dummies(dataset1.age_map,prefix='is')], axis=1)

#Now let's look at the correlation matrix to decide to retain or drop the 
#engineered features (avoiding multicollinearity).
corr = dataset1.corr()
plt.figure(figsize=(18,12))
sns.heatmap(corr, annot=True, fmt=".2g")

data = dataset1[['age', 'DebtRatio']]
corr1 = data.corr(method = 'pearson')
dataset1.columns


#Findings:

#we will retain CombinedDefaulted feature as it clearly a good predictor of our target class than the three features it was derived from.
#we will retain NumberOfTime30-59DaysPastDueNotWorse and drop the other two features derived from CombinedDefaulted as it gives a more meaningful information on our target variable (also, it looks like this is the medium range of time a borrower defaulted a payment)
#we will drop the engineered is_working and is_senior feature since age feature outperforms them.
#we will drop also the WithDependents
#we will retain CombinedCreditLoans also since it outperforms the two features it came from.
#we will drop MonthlyDebtPayments

dataset1.columns

dataset1.drop(["Unnamed: 0","NumberOfOpenCreditLinesAndLoans",\
                 "NumberOfTimes90DaysLate","NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse",\
                 "WithDependents","age_map","is_senior","is_working", "MonthlyDebtPayments"], axis=1, inplace=True)


dataset1.columns

#now let's take a look at the filtered final features to be used in predicting
#the financial distress for the next two years
corr = dataset1.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2g")

# Dataset2(test set)
def cleaned_dataset(dataset):
    dataset.loc[dataset["age"] <= 18, "age"] = dataset.age.median()
    
    age_working = dataset.loc[(dataset["age"] >= 18) & (dataset["age"] < 60)]
    age_senior = dataset.loc[(dataset["age"] >= 60)]

    age_working_impute = age_working.MonthlyIncome.mean()
    age_senior_impute = age_senior.MonthlyIncome.mean()

    dataset["MonthlyIncome"] = np.absolute(dataset["MonthlyIncome"])
    dataset["MonthlyIncome"] = dataset["MonthlyIncome"].fillna(99999)
    dataset["MonthlyIncome"] = dataset["MonthlyIncome"].astype('int64')

    dataset.loc[((dataset["age"] >= 18) & (dataset["age"] < 60)) & (dataset["MonthlyIncome"] == 99999),\
                   "MonthlyIncome"] = age_working_impute
    dataset.loc[(dataset2["age"] >= 60) & (dataset["MonthlyIncome"] == 99999), "MonthlyIncome"] = age_senior_impute
    dataset["NumberOfDependents"] = np.absolute(dataset["NumberOfDependents"])
    dataset["NumberOfDependents"] = dataset["NumberOfDependents"].fillna(0)
    dataset["NumberOfDependents"] = dataset["NumberOfDependents"].astype('int64')

    dataset["CombinedDefaulted"] = (dataset["NumberOfTimes90DaysLate"] + dataset["NumberOfTime60-89DaysPastDueNotWorse"])\
                                            + dataset["NumberOfTime30-59DaysPastDueNotWorse"]

    dataset.loc[(dataset["CombinedDefaulted"] >= 1), "CombinedDefaulted"] = 1

    dataset["CombinedCreditLoans"] = dataset["NumberOfOpenCreditLinesAndLoans"] + \
                                            dataset["NumberRealEstateLoansOrLines"]
    dataset.loc[(dataset["CombinedCreditLoans"] <= 5), "CombinedCreditLoans"] = 0
    dataset.loc[(dataset["CombinedCreditLoans"] > 5), "CombinedCreditLoans"] = 1

    dataset.drop(["Unnamed: 0","NumberOfOpenCreditLinesAndLoans",\
                 "NumberOfTimes90DaysLate","NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse"], axis=1, inplace=True)

cleaned_dataset(dataset2)  

dataset2.info()
dataset2.columns

dataset1.columns

dataset1.shape, dataset2.shape

dataset1.info()
dataset2["MonthlyIncome"] = dataset2["MonthlyIncome"].astype('int64')
dataset2["age"] = dataset2["age"].astype('int64')
dataset2.MonthlyIncome.value_counts()
dataset2.age.value_counts()
dataset1.MonthlyIncome.value_counts()
dataset2.SeriousDlqin2yrs.value_counts()
dataset1.isna().sum()
dataset2.isna().sum()
#dataset1.dtypes
#dataset2.dtypes


# split our predictors and the target variable in our datasets
X = dataset1.drop("SeriousDlqin2yrs", axis=1).copy()
y = dataset1.SeriousDlqin2yrs
X.shape, y.shape


X_test = dataset2.drop("SeriousDlqin2yrs", axis=1).copy()
y_test = dataset2.SeriousDlqin2yrs
X_test.shape, y_test.shape

#Feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#X = sc_x.fit_transform(X)
#y = sc_x.transform(y)

#Feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#X_test = sc_x.fit_transform(X_test)


##Fitting Logistic regression to the traning set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X,y)

#y_pred = classifier.predict(X_test)
#y_pred1 = classifier.predict(X)

#from sklearn.metrics import confusion_metrix
#cm = confusion_metrix()



###################
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=42)
#logit = LogisticRegression(random_state=42, solver="saga", penalty="l1", class_weight="balanced", C=1.0, max_iter=500)
#scaler = StandardScaler().fit(X_train)

#X_train.count()

#X_train_scaled = scaler.transform(X_train) #scaling features!
#X_val_scaled = scaler.transform(X_val)

#logit.fit(X_train_scaled, y_train)
#logit_scores_proba = logit.predict_proba(X_train_scaled)
#logit_scores = logit_scores_proba[:,1]

#logit.fit(X_train, y_train)
#y_pred = logit.predict(X_val)

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_val, y_pred)

#y_test_pred = logit.predict(X_test)

#########

from collections import Counter
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def print_results(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(accuracy_score(true_value, pred)))
    print("precision: {}".format(precision_score(true_value, pred)))
    print("recall: {}".format(recall_score(true_value, pred)))
    print("f1: {}".format(f1_score(true_value, pred)))
    
    
# our classifier to use
classifier = RandomForestClassifier

X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=42)

# build normal model
pipeline = make_pipeline(classifier(random_state=42))
model = pipeline.fit(X_train, y_train)
prediction = model.predict(X_val)

# build model with SMOTE imblearn
smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), classifier(random_state=42))
smote_model = smote_pipeline.fit(X_train, y_train)
smote_prediction = smote_model.predict(X_val)

# build model with undersampling
nearmiss_pipeline = make_pipeline_imb(NearMiss(random_state=42), classifier(random_state=42))
nearmiss_model = nearmiss_pipeline.fit(X_train, y_train)
nearmiss_prediction = nearmiss_model.predict(X_val)

# classification report
print(classification_report(y_val, prediction))
print(classification_report_imbalanced(y_val, smote_prediction))
print(classification_report_imbalanced(y_val, nearmiss_prediction))

print()
print('normal Pipeline Score {}'.format(pipeline.score(X_val, y_val)))
print('SMOTE Pipeline Score {}'.format(smote_pipeline.score(X_val, y_val)))
print('NearMiss Pipeline Score {}'.format(nearmiss_pipeline.score(X_val, y_val)))


print()
print_results("normal classification", y_val, prediction)
print()
print_results("SMOTE classification", y_val, smote_prediction)
print()
print_results("NearMiss classification", y_val, nearmiss_prediction)


#FOR TEST SET
# build normal model
prediction1 = model.predict(X_test)

# build model with SMOTE imblearn
smote_prediction1 = smote_model.predict(X_test)

# build model with undersampling
nearmiss_prediction = nearmiss_model.predict(X_test)


y_val.value_counts()
dataset1.columns
y_val.get_dtype_counts()

dataset1['SeriousDlqin2yrs'].unique()

dataset1['SeriousDlqin2yrs'].is_unique
#dataset1 = dataset1.set_index('SeriousDlqin2yrs')

#new_names =  {'0': 'SeriousDlqin2yrs'}

#'''Setting inplace to True specifies that our changes be made directly to the object'''
#prediction1.rename(columns= new_names, inplace= True)

'''count = 0
for i in smote_prediction:
    if i == 0:
        count+=1
    print(count)  '''
    
#most efficient frequency counts for unique values in an array
np.bincount(smote_prediction)    
dataset.corr() 


#from sklearn.metrics import r2_score
#r2_score(y_val, prediction)    
        
    