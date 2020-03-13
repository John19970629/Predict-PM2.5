# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:19:04 2019

@author: 將軍
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:17:02 2019

@author: 將軍
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:47:09 2019

@author: 將軍
"""

import pandas as pd
import numpy as np

data=pd.read_csv('train.csv')

print(data.isna().values.any())


str1='#'
str2='*'
str3='x'
str4='A'

#處理有Aㄉ值
for i in range(0,1656):
    for j in range(3,27):
        if type(data.iloc[i,j])!=float:
            if (data.iloc[i,j].find(str4)!=-1):
                data.iloc[i,j]=data.iloc[i,j].rstrip('A')
  
#把無效值都換成nan 
for i in range(0,1656):
    for j in range(3,27):
        if type(data.iloc[i,j])!=float:
            if data.iloc[i,j]!='NR':
                if (data.iloc[i,j].find(str1)!=-1 or data.iloc[i,j].find(str2)!=-1 or data.iloc[i,j].find(str3)!=-1  ):
                    data.iloc[i,j]=np.nan
                    

#把有效值轉成float     
for i in range(0,1656):
    for j in range(3,27):
        if type(data.iloc[i,j])==str:
            if data.iloc[i,j]!='NR':
                data.iloc[i,j]=float(data.iloc[i,j])
                
#把nan變成前一欄的值 把還是nan的值轉成0
data1=data.iloc[:,3:]
data1=data1.fillna(method='ffill',axis=1)    
data1=data1.fillna(value=0,axis=1)
 
#把nan變成後一欄的值 把還是nan的值轉成0
data2=data.iloc[:,3:]
data2=data2.fillna(method='bfill',axis=1)
data2=data2.fillna(value=0,axis=1)

#合併data1、data2成data3
data3=data1
for i in range(0,1656):
    for j in range(0,24):
        if data1.iloc[i,j]!='NR' and data2.iloc[i,j]!='NR':
            if data1.iloc[i,j]!=data2.iloc[i,j]:
                data3.iloc[i,j]=(data1.iloc[i,j]+data2.iloc[i,j])/2

#把data3中NR改成0 更新data
for i in range(0,1656):
    for j in range(0,24):
         if data3.iloc[i,j]=='NR':
             data3.iloc[i,j]=0
        
data.iloc[:,3:]=data3
         
#10.11月資料為訓練集 12月資料為測試集
train=data.iloc[:1098,3:27]
test=data.iloc[1098:,3:27]    
test=test.reset_index()
test=test.drop('index',axis=1)
            
#把train、test資料形式轉換為行(row)代表18種屬性，欄(column)代表逐時數據資料
train1= [([0] * 1464) for i in range(18)]

for i in range(18):#i式train1的列
    r=0 #r式train1的行
    j=i     #j式train的列
    while j<1096:
        k=0 #k式train的行
        for k in range(24):
            train1[i][r]=train.iloc[j,k]
            r=r+1
        j=j+18
        
  
    
test1= [([0] * 744) for i in range(18)]


for i in range(18):
    r=0
    j=i
    while j<558:
        k=0
        for k in range(24):
            test1[i][r]=test.iloc[j,k]
            r=r+1
        j=j+18
        
#為train_df、test_df加入features index
features=['AMB_TEMP','CH4','CO','NMHC','NO','NO2'
,'NOx'
,'O3'
,'PM10'
,'PM2.5'
,'RAINFALL'
,'RH'
,'SO2'
,'THC'
,'WD_HR'
,'WIND_DIREC'
,'WIND_SPEED'
,'WS_HR'

]     
train_df=pd.DataFrame(train1,index=features)
test_df=pd.DataFrame(test1,index=features)

######################################################################################################################################################################################

#(對train資料)取6小時為一單位切割，例如第一筆資料為第0~5小時的資料(X[0])，去預測第6小時的PM2.5值(Y[0])，下一筆資料為第1~6小時的資料(X[1])去預測第7 小時的PM2.5值(Y[1])  *hint: 切割後X的長度應為1464-6=1458
#第一種X1(取前6小時的18種特徵)
X1=[0]*1458 #1458=要被切成幾塊(每6ㄍ一塊)
r=0
for z in range(1458):
    temp=[0]*108
    i=0
    while r<1464:
        for k in range(18):
            if i<108:
                temp[i]=train_df.iloc[k,r]
                i=i+1
            else:
                break
        if i<108:
            r=r+1
        else:
            r=r-4
            break
    X1[z]=temp

  
#第一種X2(取前6小時的PM2.5特徵)
X2=[0]*1458
r=0
for z in range(1458):
    temp=[0]*6
    i=0
    while r<1464:
        if i<6:
            temp[i]=train_df.iloc[9,r]
            i=i+1
            r=r+1
        else:
            X2[z]=temp
            r=r-5
            break
            
#要被預測的第7個小時的PM2.5     
Y1=[0]*1458
r=6     #r是train_df的欄
for j in range(1458):   #j是Y的欄
    Y1[j]=train_df.iloc[9,r]
    r=r+1
    
######################################################################################################################################################################################

#(對test資料)取6小時為一單位切割，例如第一筆資料為第0~5小時的資料(X[0])，去預測第6小時的PM2.5值(Y[0])，下一筆資料為第1~6小時的資料(X[1])去預測第7 小時的PM2.5值(Y[1])  *hint: 切割後X的長度應為1464-6=1458
#第一種X3(取前6小時的18種特徵)
X3=[0]*738 
r=0
for z in range(738):
    temp=[0]*108
    i=0
    while r<744:
        for k in range(18):
            if i<108:
                temp[i]=test_df.iloc[k,r]
                i=i+1
            else:
                break
        if i<108:
            r=r+1
        else:
            r=r-4
            break
    X3[z]=temp

  
#第一種X4(取前6小時的PM2.5特徵)
X4=[0]*738
r=0
for z in range(738):
    temp=[0]*6
    i=0
    while r<744:
        if i<6:
            temp[i]=test_df.iloc[9,r]
            i=i+1
            r=r+1
        else:
            X4[z]=temp
            r=r-5
            break
            
#要被預測的第7個小時的PM2.5     
Y2=[0]*738
r=6     #r是test_df的欄
for j in range(738):   #j是Y的欄
    Y2[j]=test_df.iloc[9,r]
    r=r+1
    

######################################################################################################################################################################################

#建立Linear Regression模型
from sklearn.linear_model import LinearRegression  
LR1= LinearRegression()
LR1=LR1.fit(X1,Y1)  

LR2= LinearRegression()
LR2=LR2.fit(X2,Y1)  

#建立Random Forest Regression模型
from sklearn.ensemble import RandomForestRegressor 
RFR1=RandomForestRegressor(n_estimators = 100) 
RFR1=RFR1.fit(X1,Y1)  

RFR2=RandomForestRegressor(n_estimators = 100) 
RFR2=RFR2.fit(X2,Y1)  

######################################################################################################################################################################################
#預測，並計算MAE(4種模型 4種結果)
LR1_pred=LR1.predict(X3)
LR2_pred=LR2.predict(X4)
RFR1_pred=RFR1.predict(X3)
RFR2_pred=RFR2.predict(X4)

target=Y2
LR1_error=[]
for i in range(len(target)):
    LR1_error.append(target[i] - LR1_pred[i])
    
LR2_error=[]
for i in range(len(target)):
    LR2_error.append(target[i] - LR2_pred[i])
    
RFR1_error=[]
for i in range(len(target)):
    RFR1_error.append(target[i] - RFR1_pred[i])
    
RFR2_error=[]
for i in range(len(target)):
    RFR2_error.append(target[i] - RFR2_pred[i])

LR1_absError = []
for val in LR1_error:
    LR1_absError.append(abs(val))#误差绝对值
    
LR2_absError = []
for val in LR2_error:
    LR2_absError.append(abs(val))#误差绝对值
    
RFR1_absError = []
for val in RFR1_error:
    RFR1_absError.append(abs(val))#误差绝对值
    
RFR2_absError = []
for val in RFR2_error:
    RFR2_absError.append(abs(val))#误差绝对值
    

print("Linear Regression,18種屬性,MAE = ", sum(LR1_absError) / len(LR1_absError))
print("Linear Regression,PM2.5屬性,MAE = ", sum(LR2_absError) / len(LR2_absError))
print("Random Forest Regression,18種屬性,MAE = ", sum(RFR1_absError) / len(RFR1_absError))
print("Random Forest Regression,PM2.5屬性,MAE = ", sum(RFR2_absError) / len(RFR2_absError))

