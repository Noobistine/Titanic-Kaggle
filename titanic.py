# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 13:19:16 2016

@author: C937118
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




titanic=pd.read_csv("H:/Data/Varidas/Ausbildung AXA/Data Scientist/Kaggle/train.csv"
    , sep=',')


titanic_test=pd.read_csv("H:/Data/Varidas/Ausbildung AXA/Data Scientist/Kaggle/test.csv"
    , sep=',')
    
    
"""

First: Look at the Data

"""
  
#Scatter Plots
axes=pd.tools.plotting.scatter_matrix(
titanic[['Survived', 'Pclass','Age', 'Parch','Fare']], alpha=0.2)
plt.tight_layout()
    
#Some Information
titanic.info()
titanic.describe()

# Look at a single Record
titanic.loc[titanic['PassengerId']==5]


# Draw some Histograms
titanic['Survived'].hist(by=titanic['Sex'])
titanic['Age'].hist(by=titanic['Survived'])
titanic['Survived'].hist(by=titanic['Pclass'])
titanic['Survived'].hist(by=titanic['SibSp'])
titanic['Survived'].hist(by=titanic['Parch'])
titanic['Fare'].hist(by=titanic['Survived'])
titanic['Survived'].hist(by=titanic['Embarked'])


titanic.groupby(['Pclass','Survived']).count()


"""

Second: Calculate simple and univariate Survival Rates

"""
# Calculate survival rate for Male and Female (Univariate) 
titanic.groupby(['Sex','Survived']).count()

#Brute Force and Ignorance
female_surv=titanic.loc[(titanic['Sex']=="female") & (titanic['Survived']==1)]
male_surv=titanic.loc[(titanic['Sex']=="male") & (titanic['Survived']==1)]
female_death=titanic.loc[(titanic['Sex']=="female") & (titanic['Survived']==0)]
male_death=titanic.loc[(titanic['Sex']=="male") & (titanic['Survived']==0)]

survival_female=round(float(len(female_surv))/float(len(female_surv)+len(female_death)),2)
survival_male=round(float(len(male_surv))/float(len(male_surv)+len(male_death)),2)
print 'Surivval rate for Female is:'
print survival_female

print 'Survival rate for Male is:'
print survival_male

#And then the nice way
titanic.groupby(['Sex']).mean()['Survived']


#Calculate survival Rate dependent on Age

titanic['Age_grp']=np.round(titanic['Age']/5)*5
survival_age1=titanic['Age_grp'].value_counts()
survival_age2=titanic.groupby('Age_grp')['Survived'].sum()

survival_age=pd.concat([survival_age1, survival_age2], axis=1)

survival_age['survival_rate']=survival_age['Survived']/survival_age['Age_grp']
survival_age.reset_index(level=0, inplace=True)

survival_age.plot(x='index', y='survival_rate')

ax=survival_age[['index', 'survival_rate']].plot(x='index', linestyle='-')
survival_age[['index', 'Age_grp']].plot(x='index', kind='bar')

# Check also Survival Rate dependent ond Embarked
titanic.groupby(['Embarked']).mean()['Survived']

"""

Three: Clean Data

"""

# Clean Data

#Insert missing Age

#Check if the other Information is also missing, if so then delete these records
null_data=titanic[titanic.isnull().Age]
#--> The rest is known, so insert missing Age with median

#Calculate median Age per Parch and SibSp Group
help_median=titanic.groupby(['Parch', 'SibSp']).median()['Age'].to_frame()

#Add Index to the dataframe as a string variable to later merge with train
help_median['key']=help_median.index.map(str)

# Ad Key Variable to merge with median Age
titanic['key']="("+titanic['Parch'].map(str)+"L, "+titanic['SibSp'].map(str)+"L)"

#Merge
titanic2=pd.merge(titanic, help_median, how='left', on='key')
titanic2['Age_clean']=np.where(titanic2['Age_x']>=0, titanic2['Age_x'], titanic2['Age_y'])


#There are still some missing
titanic2[titanic2.isnull().Age_clean]
titanic2['Age_clean']=titanic2['Age_clean'].fillna(titanic2['Age_clean'].median())
titanic2[titanic2.isnull().Age_clean]

# Map Gender to boolean Variable
titanic2['Gender']=np.where(titanic2['Sex']=='female',0,1)

#Map Embarked to integer
titanic2['Embarked'].unique()
titanic2['Embarked_int']=np.where(titanic2['Embarked']=="C",1,np.where(titanic2['Embarked']=="Q",2,0))



"""

Four: Try some Models

"""

"""

Linear Regression

"""
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

predictors=['Pclass', 'Gender', 'Age_clean', 'SibSp', 'Parch', 'Fare', 'Embarked_int'] 

cross_val=KFold(titanic2.shape[0], n_folds=3, random_state=1)

#Initialize Series to store prediction results
predictions=[]
alg=LinearRegression()

    
for train, test in cross_val:
    train_pred=titanic2[predictors].iloc[train,:]
    train_response=titanic2['Survived'].iloc[train]
    
    alg.fit(train_pred, train_response)
    
    test_predictions=alg.predict(titanic2[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
predictions=np.concatenate(predictions, axis=0)
pred=pd.DataFrame(predictions)

pred['out']=np.where(pred[0]<0.5,0,1)

titanic3=pd.concat([titanic2,pred], axis=1)
titanic3.describe()
titanic3[0].hist() #Values from -0.3 to +1.2

titanic3['error']=np.where(titanic3['Survived']==titanic3['out'],1,0)

blub=titanic3.groupby(titanic3['error']).count()

accuracy_lin=float(blub.iloc[1][0])/float(len(titanic3))



"""

Logistic Regression

"""

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

alg=LogisticRegression(random_state=1)

scores=cross_validation.cross_val_score(alg,titanic2[predictors],
                                        titanic['Survived'], cv=3)
                                        
accuracy_logit=scores.mean()
                                       
alg.fit(titanic2[predictors], titanic2['Survived'])



#Clean titanic_test likewise
titanic_test['key']="(" + titanic_test['Parch'].map(str)+"L, "+titanic_test['SibSp'].map(str)+"L)"

titanic_test2=pd.merge(titanic_test, help_median, how='left', on='key')
titanic_test2['Age_clean']=np.where(titanic_test2['Age_x']>=0,
                                    titanic_test2['Age_x'],
                                    titanic_test2['Age_y'])
titanic_test2['Age_clean']=titanic_test2['Age_clean'].fillna(titanic['Age'].median())

titanic_test2['Gender']=np.where(titanic_test2['Sex']=="female",0,1)
titanic_test2['Embarked_int']=np.where(titanic_test2['Embarked']=="Q",2,
                                        np.where(titanic_test2['Embarked']=="C",1,0))

titanic_test2[titanic_test2.isnull().Fare]
titanic_test2['Fare']=titanic_test2['Fare'].fillna(titanic_test2['Fare'].median())


predictions=alg.predict(titanic_test2[predictors])
submission=pd.DataFrame({'PassengerId': titanic_test['PassengerId'], 
                         'Survived': predictions})


"""

Random Forrest

"""

from sklearn.ensemble import RandomForestClassifier

alg=RandomForestClassifier(random_state=1, 
                           n_estimators=50, #Number of Trees
                           min_samples_split=8, #Minimum number of Rows
                           min_samples_leaf=4) #Minium Samples at the bottom of the tree

cross_val=KFold(titanic2.shape[0], n_folds=3, random_state=1)

scores=cross_validation.cross_val_score(alg,titanic2[predictors], 
                                        titanic2['Survived'], cv=cross_val)

accuracy_rf=scores.mean()


"""

Gradient Boosting

"""

from sklearn.ensemble import GradientBoostingClassifier

alg=GradientBoostingClassifier(random_state=1,
                               n_estimators=20, #Number of Trees
                               max_depth=4)
                             
scores=cross_validation.cross_val_score(alg, titanic2[predictors], titanic2['Survived'], cv=cross_val)
                               
accuracy_gb=scores.mean()

alg.fit(titanic2[predictors], titanic2['Survived'])

prediction=alg.predict(titanic_test2[predictors])

submission=pd.DataFrame({'PassengerId': titanic_test2['PassengerId'],
                         'Survived': prediction})
                         
submission.to_csv("H:/Data/Varidas/Ausbildung AXA/Data Scientist/Kaggle/submission.csv", index=False)