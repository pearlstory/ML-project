import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression,Lasso,Ridge,BayesianRidge
import Predict_profit
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

#some function for predicting
def Predict_schooling_logistic(train,test,kind):
    n=test.shape[1]
    train=pd.get_dummies(train)
    if n==17:
        for i in range(17,25):
            logistic_model = LogisticRegression()
            logistic_model.fit(train[train.columns[0:17]],train[train.columns[i]])
            fitted_test = logistic_model.predict_proba(test[test.columns[0:17]])[:, 1]
            if i==17:
                r=fitted_test
            else:
                r=np.vstack((r,fitted_test))
        r=r.T
        for j in range(0,r.shape[0]):
            maxp=max(r[j])
            r[j]=r[j]-maxp
            r[j][r[j]>=0]=1
            r[j][r[j]<0]=0
            r_list=r[j].tolist()
            if j==0:
                Pred_schooling=[kind[r_list.index(1)]]
            else:
                Pred_schooling.append(kind[r_list.index(1)])
        Pred_schooling_index=test.index.tolist()
        return Pred_schooling_index,Pred_schooling
    else:
        for i in range(17,25):
            logistic_model = LogisticRegression()
            logistic_model.fit(train[train.columns[1:17]],train[train.columns[i]])
            fitted_test = logistic_model.predict_proba(test)[:, 1]
            if i==17:
                r=fitted_test
            else:
                r=np.vstack((r,fitted_test))
        r=r.T
        for j in range(0,r.shape[0]):
            maxp=max(r[j])
            r[j]=r[j]-maxp
            r[j][r[j]>=0]=1
            r[j][r[j]<0]=0
            r_list=r[j].tolist()
            if j==0:
                Pred_schooling=[kind[r_list.index(1)]]
            else:
                Pred_schooling.append(kind[r_list.index(1)])
        Pred_schooling_index=test.index.tolist()
        return Pred_schooling_index,Pred_schooling
    
def Predict_age_ridge(train,test):
    n=test.shape[1]
    pred_index=test.index
    if n==16:
        y_train=train['custAge']
        X_train=pd.get_dummies(train.drop(columns=['custAge','schooling']))
        clf = Ridge(alpha=1)
        clf.fit(X_train,y_train)
        pred=clf.predict(test)
        pred=pred.round()
        return pred_index,pred
    else:
        y_train=train['custAge']
        X_train=pd.get_dummies(train.drop(columns=['custAge']))
        clf = Ridge(alpha=1)
        clf.fit(X_train,y_train)
        pred=clf.predict(test)
        pred=pred.round()
        return pred_index,pred



def Final_predict_real(test_df_dum,X_all,y_responded,X_profit,y_profit,opt,accuracy):
    X_all=Predict_profit.judge_process(X_all)
    X_profit=Predict_profit.judge_process(X_profit)
    test_df_dum=Predict_profit.judge_process(test_df_dum)
    if opt==0:#RF
        randomforest = RandomForestClassifier(n_estimators=1000,n_jobs=8,max_depth=10)
        randomforest.fit(X_all,y_responded)
        pred_responded=randomforest.predict(test_df_dum)
    elif opt==1:#GBDT
        GBDT = GradientBoostingClassifier(learning_rate=0.1,n_estimators=200)  
        GBDT.fit(X_all,y_responded)
        pred_responded=GBDT.predict(test_df_dum) 
    elif opt==2:#DT
        DT = tree.DecisionTreeClassifier()
        DT.fit(X_all,y_responded)
        pred_responded=DT.predict(test_df_dum)
    elif opt==3:#KNN
        KNN=KNeighborsClassifier()
        KNN.fit(X_all,y_responded)
        pred_responded=KNN.predict(test_df_dum)  
    elif opt==4:
        logreg = LogisticRegression()
        logreg.fit(X_all,y_responded)
        pred_responded=logreg.predict(test_df_dum)
    lasso = Lasso(alpha= 0.1)
    lasso.fit(X_profit,y_profit)
    pred_profit=lasso.predict(test_df_dum)

    X_all_0 = X_all.loc[(y_responded == 0)]
    #pred_profit_0=ridge.predict(X_all_0)/(X_all_0.shape[0])
    realprofit=np.zeros(test_df_dum.shape[0])
    marketlabel=np.ones(test_df_dum.shape[0])
    for i in range(0,test_df_dum.shape[0]):
        if pred_responded[i]==0:
            realprofit[i]=pred_profit[i]*(1-accuracy)+accuracy*(-30)
        elif pred_responded[i]==1:
            realprofit[i]=pred_profit[i]*accuracy+(1-accuracy)*(-30)
        if realprofit[i]<0:
            marketlabel[i]=0
    #pred_profit[ID_no]=0
    #marketlabel=np.ones(test_df_dum.shape[0])
    #marketlabel[ID_no]=0
    #marketlabel[pred_profit<30]=0
    test_df = pd.read_csv('DataPredict.csv')
    col_name = test_df.columns.tolist()
    test_df.insert(col_name.index('pastEmail')+1,'responded',pred_responded)
    col_name = test_df.columns.tolist()
    test_df.insert(col_name.index('responded')+1,'profit',pred_profit)
    col_name = test_df.columns.tolist()
    test_df.insert(col_name.index('profit')+1,'market',marketlabel)
    ID_no=test_df_dum.iloc[np.where(marketlabel==0)[0]].index.tolist()
    pred_profit[ID_no]=0
    #realprofit=sum(pred_profit)-sum(responded)*30-sum(responded)*(1-accuracy)*175.18-(930-sum(responded))*(1-accuracy)*145
    totalprofit=sum(realprofit)-sum(marketlabel)*30
    return test_df,totalprofit

def Final_predict(test_df_dum,X_all,y_responded,X_profit,y_profit,opt,accuracy):
    X_all=Predict_profit.judge_process(X_all)
    X_profit=Predict_profit.judge_process(X_profit)
    test_df_dum=Predict_profit.judge_process(test_df_dum)
    if opt==0:#RF
        randomforest = RandomForestClassifier(n_estimators=1000,n_jobs=8,max_depth=10)
        randomforest.fit(X_all,y_responded)
        pred_responded=randomforest.predict(test_df_dum)
    elif opt==1:#GBDT
        GBDT = GradientBoostingClassifier(learning_rate=0.1,n_estimators=200)  
        GBDT.fit(X_all,y_responded)
        pred_responded=GBDT.predict(test_df_dum) 
    elif opt==2:#DT
        DT = tree.DecisionTreeClassifier()
        DT.fit(X_all,y_responded)
        pred_responded=DT.predict(test_df_dum)
    elif opt==3:#KNN
        KNN=KNeighborsClassifier()
        KNN.fit(X_all,y_responded)
        pred_responded=KNN.predict(test_df_dum)  
    elif opt==4:
        logreg = LogisticRegression()
        logreg.fit(X_all,y_responded)
        pred_responded=logreg.predict(test_df_dum)
    lasso = Lasso(alpha= 0.1)
    lasso.fit(X_profit,y_profit)
    pred_profit=lasso.predict(test_df_dum).round(0)

    X_all_0 = X_all.loc[(y_responded == 0)]
    #pred_profit_0=ridge.predict(X_all_0)/(X_all_0.shape[0])
    realprofit=np.zeros(test_df_dum.shape[0])
    marketlabel=np.ones(test_df_dum.shape[0])
    ID_no=test_df_dum.iloc[np.where(pred_responded==0)[0]].index.tolist()
    #pred_profit[ID_no]=0
    #marketlabel=np.ones(test_df_dum.shape[0])
    marketlabel[ID_no]=0
    marketlabel[pred_profit<30]=0
    test_df = pd.read_csv('DataPredict.csv')
    col_name = test_df.columns.tolist()
    test_df.insert(col_name.index('pastEmail')+1,'responded',pred_responded)
    col_name = test_df.columns.tolist()
    test_df.insert(col_name.index('responded')+1,'profit',pred_profit)
    col_name = test_df.columns.tolist()
    test_df.insert(col_name.index('profit')+1,'market',marketlabel)
    ID_no=test_df_dum.iloc[np.where(marketlabel==0)[0]].index.tolist()    
    pred_profit[ID_no]=0
    #realprofit=sum(pred_profit)-sum(responded)*30-sum(responded)*(1-accuracy)*175.18-(930-sum(responded))*(1-accuracy)*145
    totalprofit=sum(pred_profit)-sum(marketlabel)*30
    return test_df,totalprofit
