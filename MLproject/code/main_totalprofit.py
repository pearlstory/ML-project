
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import re as re
get_ipython().magic(u'matplotlib inline')
import Getmissed
import Processothers
import Addmissingvalue
from sklearn.linear_model import LogisticRegression
import Addpredict
import Predict
from sklearn.linear_model import Ridge
from imblearn.over_sampling import SMOTE
import Predict_Estimation
from sklearn.model_selection import cross_val_score,train_test_split
import Predict_profit
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns

#read data
train_df = pd.read_csv('DataTraining.csv')
test_df = pd.read_csv('DataPredict.csv')
#fill missing data
train=Getmissed.GetTrain(train_df)
train_df=Processothers.Get_processed_and_complete(train_df,train)
#get data for profit
[X_profit,y_profit]=Processothers.Get_processed_profit_need_data(train_df)
#SMOTE
train_df=Processothers.Process_SMOTE(train_df)
#get data for responded
[X_all,y_all]=Processothers.Process_get_Xdata_y(train_df,0)
#fill missing data for test dataset
test_df=Processothers.Get_processed_and_complete(test_df,train)
test_df=Processothers.Process_get_nice_test_df(test_df,train_df)
#normalise
test_df=Predict_profit.judge_process(test_df)
X_all=Predict_profit.judge_process(X_all)
X_profit=Predict_profit.judge_process(X_profit)
#get final CSV and totalprofit
test_df_GBDT_good,totalptofit_GBDT=Predict.Final_predict(test_df,X_all,y_all,X_profit,y_profit,1,0.9476)
test_df_GBDT_good.to_csv("DataPredict.csv",index=False,sep=',')

