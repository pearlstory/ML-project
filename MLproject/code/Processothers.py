import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from imblearn.over_sampling import SMOTE 
import Getmissed
import Addmissingvalue
import Addpredict
import Predict
import matplotlib.pyplot as plt
import re as re
#import Processothers

#some function for processing the data
def Process_dayofweek(train_df):
    train_df['day_of_week']=train_df['day_of_week'].fillna(method='pad')
    return train_df

def Process_campaign(train_df):
    value_of_campaign=train_df['campaign']
    value_of_campaign=value_of_campaign.tolist()
    collist_of_train_df=train_df.columns.tolist()
    campaign_index=collist_of_train_df.index('campaign')
    train_df.insert(campaign_index+1,'campaign_days',value_of_campaign)
    notdays_df=train_df[train_df.campaign_days<=10]
    notdays_index=notdays_df.index
    train_df.loc[notdays_index,'campaign_days']=0  
    nottimes_df=train_df[train_df.campaign>10]
    nottimes_index=nottimes_df.index
    train_df.loc[nottimes_index,'campaign']=0
    return train_df

def Process_pdays_pmonths(train_df):
    pdays999_df=train_df[train_df.pdays==999]
    pdays_index=pdays999_df.index
    train_df.loc[pdays_index,'pdays']=0
    pmonths999_df=train_df[train_df.pmonths==999]
    pmonths_index=pmonths999_df.index
    train_df.loc[pmonths_index,'pmonths']=0
    return train_df

def Process_get_class(kind):
    kind=kind.drop_duplicates()
    kind=kind.tolist()
    kind=sorted(kind)
    return kind

def Process_responded(train_df):
    yes_index= train_df[train_df.responded=='yes'].index
    train_df.loc[yes_index,['responded']]=1
    no_index= train_df[train_df.responded=='no'].index   
    train_df.loc[no_index,['responded']]=0
    return train_df

def Process_age_round(train_df):
    train_df['custAge']=train_df['custAge'].round()
    return train_df

def Process_profit_round(train_df):
    train_df['profit']=train_df['profit'].round()
    return train_df

def Process_SMOTE(train_df):
    olabel=train_df['responded']
    odata=train_df.drop(columns=['id','responded'])
    odata=pd.get_dummies(odata)
    odata['profit'].fillna(0, inplace=True)
    X_resampled, y_resampled = SMOTE(kind='borderline2',random_state=27).fit_sample(odata, olabel)
    oindex=range(0,len(y_resampled))
    df=DataFrame(X_resampled,index=oindex,columns=odata.columns)
    df['profit'].replace(0,np.nan)
    df=Process_age_round(df)
    df=Process_profit_round(df)
    y_profit=df['profit']
    df=df.drop(columns=['profit'])
    df_col=df.columns.tolist()
    df.insert(df_col.index('poutcome_success')+1,'responded',y_resampled)
    df_col=df.columns.tolist()
    df.insert(df_col.index('responded')+1,'profit',y_profit)
    return df

#input:dum not include responded\profit\id   coef
def Process_del_zero_feature(X_data_dum,coef):
    n1=X_data_dum.shape[1]
    n2=len(coef)
    if n1==n2:
        drop_weit = np.where(coef==0)
        drop_feature = X_data_dum.columns[drop_weit]
        remain = X_data_dum.drop(drop_feature,axis=1, inplace=False)
        return remain
    else:
        print 'The data and the coefficient is not euqal!!!'
        return 0
    
#input:train_df_dum,include responded and profit, not include id
#option: 0 is responded ,1 is profit
def Process_get_Xdata_y(train_df_dum,opt):
    if opt==0:
        olabel=train_df_dum['responded']
        odata=train_df_dum.drop(columns=['responded','profit'])
        return odata,olabel
    if opt==1:
        olabel=train_df_dum['profit']
        odata=train_df_dum.drop(columns=['responded','profit'])
        odata=odata.loc[(olabel != 0)]
        olabel= olabel[(olabel !=0)]
        return odata,olabel

def Process_get_nice_test_df(test_df,train_df):
    test_df=pd.get_dummies(test_df)
    train_df_col=train_df.columns.tolist()
    test_df_col=test_df.columns.tolist()
    for kind in test_df_col:
        train_df_col.remove(kind)
    lost_col = train_df_col
    train_df_col=train_df.columns.tolist()
    for losskind in lost_col:
        addval=np.zeros(test_df.shape[0])
        addindex=train_df_col.index(losskind)
        test_df.insert(addindex,losskind,addval)
    test_df=test_df.drop(columns=['profit','responded'])   
    return test_df

#train_df = pd.read_csv('DataTraining.csv')
#train=Getmissed.GetTrain(train_df)
def Get_processed_and_complete(train_df,train):
    kind=Process_get_class(train['schooling'])
    #loss age
    Agemiss=Getmissed.GetAgemissed(train_df)
    Agemissdum=Addmissingvalue.AddAge_missedval(train,Agemiss)
    [index,value]=Predict.Predict_age_ridge(train,Agemissdum)
    train_df=Addpredict.Add_age(train_df,value,index)
    #loss schooling
    Schoolingmiss=Getmissed.GetSchoolingmissed(train_df)
    Schoolingmissdum=Addmissingvalue.AddSchooling_missedval(train,Schoolingmiss)
    [index,value]=Predict.Predict_schooling_logistic(train,Schoolingmissdum,kind)
    train_df=Addpredict.Add_schooling(train_df,value,index)
    #loss age and school
    AgeSchoolingmiss=Getmissed.GetAgeSchoolingmissed(train_df)
    AgeSchoolingmissdum=Addmissingvalue.AddAgeSchooling_missedval(train,AgeSchoolingmiss)
    [age_index,age_value]=Predict.Predict_age_ridge(train,AgeSchoolingmissdum)
    train_df=Addpredict.Add_age(train_df,age_value,age_index)
    [schooling_index,schooling_value]=Predict.Predict_schooling_logistic(train,AgeSchoolingmissdum,kind)
    train_df=Addpredict.Add_schooling(train_df,schooling_value,schooling_index)
    #process
    train_df=Process_campaign(train_df)
    train_df=Process_pdays_pmonths(train_df)
    train_df=Process_dayofweek(train_df)
    if len(train_df.columns)==25:
        train_df=Process_responded(train_df)
    return train_df

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
def get_importance_of_feature(X_all,y_all):
    clf = ExtraTreesClassifier()
    clf = clf.fit(X_all, y_all)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X_all)
    dc=pd.DataFrame(clf.feature_importances_) 
    dc.columns=['importance']
    col_name = dc.columns.tolist()
    dc.insert(col_name.index('importance')+1,'feature',X_all.columns)
    dc=dc.sort_values(by='importance', ascending=False)
    sns.set_style("whitegrid")

    plt.figure(figsize=(19, 25),dpi=500)
    ax = sns.barplot(x="importance", y="feature", data=dc)
    foo_fig = plt.gcf() 
    foo_fig.savefig('snsfull.eps', format='eps', dpi=1200)
    plt.show() 
    
    dcp=dc.loc[(dc.importance >= 0.02)]
    plt.figure(figsize=(20, 15),dpi=500)
    ax = sns.barplot(x="importance", y="feature", data=dcp)
    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig('snspart.eps', format='eps', dpi=1200)
    plt.show() 
    return dc

def Get_processed_profit_need_data(train_df):
    olabel=train_df['profit']
    y_responded=train_df['responded']
    odata=train_df.drop(columns=['id','responded','profit'])
    odata=pd.get_dummies(odata)
    odata=odata.loc[(y_responded != 0)]
    olabel = olabel[(y_responded !=0)]
    return odata,olabel