import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import Ridge
from imblearn.over_sampling import SMOTE
from sklearn.cross_validation import cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
import Predict_profit
#test the performance of classification algorithm
def Predict_evaluation_classify(train_df):
    df=train_df
    df=df.sample(frac=1).reset_index(drop=True)
    olabel=df['responded']
    odata=df.drop(columns=['responded','profit'])
    X_all=odata
    y_all=olabel
    X_all=Predict_profit.judge_process(X_all)
    #logistic
    logreg = LogisticRegression()
    logreg_score = cross_val_score(logreg, X_all, y_all, cv=10, scoring='accuracy')
    print 'The accuracy by using LogisticRegression is:', logreg_score.mean().round(4)
    #random forest
    randomforest = RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=8,max_depth=10)
    randomforest_score = cross_val_score(randomforest, X_all, y_all, cv=10, scoring='accuracy')
    print 'The accuracy by using RandomForestClassifier is:', randomforest_score.round(4)
    #SVM
    #Svm = svm.SVC()
    #Svm_score = cross_val_score(Svm, X_all, y_all, cv=10, scoring='accuracy')
    #print 'The accuracy by using SVM is:', Svm_score.mean().round(4)
    #decision tree
    DT = tree.DecisionTreeClassifier()
    DT_score = cross_val_score(DT, X_all, y_all, cv=10, scoring='accuracy')
    print 'The accuracy by using Decision Tree is:', DT_score.mean().round(4)
    #KNN
    KNN=KNeighborsClassifier()
    KNN_score = cross_val_score(KNN, X_all, y_all, cv=10, scoring='accuracy')
    print 'The accuracy by using K neighbors is:', KNN_score.mean().round(4)
    #Bayes
    Bayes = GaussianNB()
    Bayes_score = cross_val_score(Bayes, X_all, y_all, cv=10, scoring='accuracy')
    print 'The accuracy by using Naive Bayes is:', Bayes_score.mean().round(4)
    #GBDT(Gradient Boosting Decision Tree) Classifier  
    GBDT = GradientBoostingClassifier(learning_rate=0.1,n_estimators=200)   
    GBDT_score = cross_val_score(GBDT, X_all, y_all, cv=10, scoring='accuracy')
    print 'The accuracy by using Gradient Boosting Decision Tree is:', GBDT_score.round(4)

    return 0

