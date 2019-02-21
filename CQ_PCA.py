
# coding: utf-8

# In[6]:


import os
os.getcwd()


# In[1]:


import pandas as pd
df_ori = pd.read_csv('data\\dw_cl_jl_high_value.csv',delimiter='|')



category_cols=['CITY_DESC','DESC_1ST','SYSTEM_FLAG','PRODUCT_TYPE','RH_TYPE',
           'FLUX_RELEASE','DX_FLUX','DX_EFF_DATE','SUB_PURCH_MODE_TYPE','PAYMENT',]

#data_conti= df_ori[conti_cols]
#data_cate= df_ori[category_cols]
df_user = pd.read_csv('data\\lost_user_7in8not.csv')

data=pd.merge(df_user,df_ori,how='left',left_on='USER_ID_7in8not',right_on='USER_ID')

len(data)


# In[2]:


data_cate= data[category_cols]


# In[3]:



import time
import numpy as np

def f(x):
    if str(x)=='00000000':
        return x
    elif  ' ' in str(x):
        ymd = str(x).split(' ')[0].strip()
    else:
        ymd = str(x).strip()
    return  time.strftime('%Y%m%d',time.strptime(ymd,'%Y/%m/%d'))

data_cate['DX_EFF_DATE'] = data['DX_EFF_DATE'].fillna('00000000')

data_cate['DX_EFF_DATE'] = data['DX_EFF_DATE'].map(lambda x: str(x).replace('nan','00000000')).map(f)
data_cate['DX_EFF_DATE'].value_counts()
    


# In[4]:


data_cate['CITY_DESC']=data['CITY_DESC'].fillna('未知').astype('str')
data_cate['CITY_DESC']=data['CITY_DESC'].map(lambda x:str(x).replace('主城未知','未知').replace('nan','未知'))
data_cate['CITY_DESC'].value_counts()


# In[5]:


#############deal_status():
def fun(x):
    if('停'in str(x)):
        return '停机'
    if('拆机' in str(x)):
        return '拆机'
    else:
        return x
data_cate['DESC_1ST'] = data['DESC_1ST'].fillna('未知').astype('str')
data_cate['DESC_1ST'] = data['DESC_1ST'].map(fun)
print(data_cate['DESC_1ST'].value_counts())
print("\r===========================================\r")

#eal_release():
data_cate['FLUX_RELEASE']=data['FLUX_RELEASE'].fillna('未释放').astype('str')
data_cate['FLUX_RELEASE'] = data['FLUX_RELEASE'].map(lambda x:str(x).replace('nan','未释放').replace('CBSS','').replace('BSS',''))

data_cate['FLUX_RELEASE'].value_counts()
#for str_col in category_cols:
#    tmp=pd.Series(data_cate[str_col])
#    print(str_col+" ================: \r")
#    print(tmp.value_counts())


# In[6]:


###### 融合套餐
data_cate['RH_TYPE']=data_cate['RH_TYPE'].fillna('非融合').astype('str')
data_cate['RH_TYPE'].value_counts()


# In[7]:


data_cate['SYSTEM_FLAG'].value_counts()


# In[8]:


data_cate['PRODUCT_TYPE']=data_cate['PRODUCT_TYPE'].fillna('其他')
data_cate['PRODUCT_TYPE'].value_counts()


# In[9]:


data_cate['SUB_PURCH_MODE_TYPE'] =data_cate['SUB_PURCH_MODE_TYPE'].fillna('其他')
data_cate['SUB_PURCH_MODE_TYPE'].value_counts()


# In[10]:


data_cate['PAYMENT'] =data_cate['PAYMENT'].fillna('其他')
data_cate['PAYMENT'].value_counts()


# In[11]:


conti_cols=['INNET_DATE','PRODUCT_FEE'  #,'TOTAL_FLUX_THIS','FEE_THIS',,'CALLING_DURA_THIS','CHARGE_FLUX_08','CHARGE_VOICE_08','INNER_ROAM_FEE_08'
,'PROD_IN_VOICE','PROD_IN_FLUX','CALL_NUM',
'IS_TERM','TERM_END_MONTH','IS_LH','LH_END_MONTH'
,'DX_FEE','DX_END_DATE','DX_HIGH_SPEED','ARPU_AVG','FEE_LAST1','FEE_LAST2','FEE_LAST3'
,'FEE_LAST4','FEE_LAST5','CALLING_DURA_AVG_3'
,'CALLING_DURA_LAST1','CALLING_DURA_LAST2','CALLING_DURA_LAST3'
,'CALLING_DURA_LAST4','CALLING_DURA_LAST5','AVG_FLUX_3'
,'TOTAL_FLUX_LAST1','TOTAL_FLUX_LAST2','TOTAL_FLUX_LAST3'
,'TOTAL_FLUX_LAST4','TOTAL_FLUX_LAST5','EX_FLUX_FEE_THIS'
,'EX_FLUX_FEE_LAST1','EX_FLUX_FEE_LAST2','EX_FLUX_FEE_LAST3'
,'EX_FLUX_FEE_LAST4','EX_FLUX_FEE_LAST5','PKG_FLUX_FEE_THIS'
,'PKG_FLUX_FEE_LAST1','PKG_FLUX_FEE_LAST2','PKG_FLUX_FEE_LAST3'
,'PKG_FLUX_FEE_LAST4','PKG_FLUX_FEE_LAST5','VOICE_FEE_THIS'
,'VOICE_FEE_LAST1','VOICE_FEE_LAST2','VOICE_FEE_LAST3','VOICE_FEE_LAST4'
,'VOICE_FEE_LAST5','CHARGE_FLUX_07'
,'CHARGE_VOICE_07','HAS_FK','HAS_ADSL','WXBD','STBD','OPPOSE_LINK'
,'CDR_KF_OUT','CDR_KF_IN','IS_YH','YH_NUM'
,'INNER_ROAM_FEE_07','INNER_ROAM_FEE_06','IS_FP_PRINT','PRINT_CNT'
,'IS_TS','TS_CNT','IS_VIDEO','IS_CHANGE'
,'IS_GWLS'
,'RELEASE_FLAG','OVER_FLUX_FEE_AVG','YY_FLAG','BUILDING_INFO'
,'OVER_VOICE_FEE_AVG','IS_ZK']

data_conti= data[conti_cols]

data_conti['FEE_LAST1-2']=(data_conti['FEE_LAST1']-data_conti['FEE_LAST2']) / ( data_conti['FEE_LAST2'] + 1)
data_conti['FEE_LAST2-3']=(data_conti['FEE_LAST2']-data_conti['FEE_LAST3']) / (data_conti['FEE_LAST3'] + 1 )
data_conti['FEE_LAST3-4']=(data_conti['FEE_LAST3']-data_conti['FEE_LAST4']) / (data_conti['FEE_LAST4'] + 1 )
data_conti['FEE_LAST4-5']=(data_conti['FEE_LAST4']-data_conti['FEE_LAST5']) / (data_conti['FEE_LAST5'] + 1 )

data_conti['CALLING_DURA_LAST1-2']=(data_conti['CALLING_DURA_LAST1']-data_conti['CALLING_DURA_LAST2'])/(data_conti['CALLING_DURA_LAST2']+1)
data_conti['CALLING_DURA_LAST2-3']=(data_conti['CALLING_DURA_LAST2']-data_conti['CALLING_DURA_LAST3'])/(data_conti['CALLING_DURA_LAST3']+1)
data_conti['CALLING_DURA_LAST3-4']=(data_conti['CALLING_DURA_LAST3']-data_conti['CALLING_DURA_LAST4'])/(data_conti['CALLING_DURA_LAST4']+1)
data_conti['CALLING_DURA_LAST4-5']=(data_conti['CALLING_DURA_LAST4']-data_conti['CALLING_DURA_LAST5'])/(data_conti['CALLING_DURA_LAST5']+1)

data_conti['TOTAL_FLUX_LAST1-2']=(data_conti['TOTAL_FLUX_LAST1']-data_conti['TOTAL_FLUX_LAST2'])/(data_conti['TOTAL_FLUX_LAST2']+1)
data_conti['TOTAL_FLUX_LAST2-3']=(data_conti['TOTAL_FLUX_LAST2']-data_conti['TOTAL_FLUX_LAST3'])/(data_conti['TOTAL_FLUX_LAST3']+1)
data_conti['TOTAL_FLUX_LAST3-4']=(data_conti['TOTAL_FLUX_LAST3']-data_conti['TOTAL_FLUX_LAST4'])/(data_conti['TOTAL_FLUX_LAST4']+1)
data_conti['TOTAL_FLUX_LAST4-5']=(data_conti['TOTAL_FLUX_LAST4']-data_conti['TOTAL_FLUX_LAST5'])/(data_conti['TOTAL_FLUX_LAST5']+1)

data_conti['EX_FLUX_FEE_LAST1-2']=(data_conti['EX_FLUX_FEE_LAST1']-data_conti['EX_FLUX_FEE_LAST2'])/(data_conti['EX_FLUX_FEE_LAST2']+1)
data_conti['EX_FLUX_FEE_LAST2-3']=(data_conti['EX_FLUX_FEE_LAST2']-data_conti['EX_FLUX_FEE_LAST3'])/(data_conti['EX_FLUX_FEE_LAST3']+1)
data_conti['EX_FLUX_FEE_LAST3-4']=(data_conti['EX_FLUX_FEE_LAST3']-data_conti['EX_FLUX_FEE_LAST4'])/(data_conti['EX_FLUX_FEE_LAST4']+1)
data_conti['EX_FLUX_FEE_LAST4-5']=(data_conti['EX_FLUX_FEE_LAST4']-data_conti['EX_FLUX_FEE_LAST5'])/(data_conti['EX_FLUX_FEE_LAST5']+1)

data_conti['PKG_FLUX_FEE_LAST1-2']=(data_conti['PKG_FLUX_FEE_LAST1']-data_conti['PKG_FLUX_FEE_LAST2'])/(data_conti['PKG_FLUX_FEE_LAST2']+1)
data_conti['PKG_FLUX_FEE_LAST2-3']=(data_conti['PKG_FLUX_FEE_LAST2']-data_conti['PKG_FLUX_FEE_LAST3'])/(data_conti['PKG_FLUX_FEE_LAST3']+1)
data_conti['PKG_FLUX_FEE_LAST3-4']=(data_conti['PKG_FLUX_FEE_LAST3']-data_conti['PKG_FLUX_FEE_LAST4'])/(data_conti['PKG_FLUX_FEE_LAST4']+1)
data_conti['PKG_FLUX_FEE_LAST4-5']=(data_conti['PKG_FLUX_FEE_LAST4']-data_conti['PKG_FLUX_FEE_LAST5'])/(data_conti['PKG_FLUX_FEE_LAST5']+1)

data_conti['VOICE_FEE_LAST1-2']=(data_conti['VOICE_FEE_LAST1']-data_conti['VOICE_FEE_LAST2'])/(data_conti['VOICE_FEE_LAST2']+1)
data_conti['VOICE_FEE_LAST2-3']=(data_conti['VOICE_FEE_LAST2']-data_conti['VOICE_FEE_LAST3'])/(data_conti['VOICE_FEE_LAST3']+1)
data_conti['VOICE_FEE_LAST3-4']=(data_conti['VOICE_FEE_LAST3']-data_conti['VOICE_FEE_LAST4'])/(data_conti['VOICE_FEE_LAST4']+1)
data_conti['VOICE_FEE_LAST4-5']=(data_conti['VOICE_FEE_LAST4']-data_conti['VOICE_FEE_LAST5'])/(data_conti['VOICE_FEE_LAST5']+1)


# In[12]:


data_cate['DX_FLUX'].value_counts()
dx_flux_dic={
    '1.5GB本地流量':1.5,
    '1.5G本地流量':1.5,
    '100G本地流量':100,
    '10GB本地流量':10,
    '10G本地流量':10,
    '15G本地流量':10,
    '1GB本地流量':1,
    '1G本地流量':1,
    '20GB本地+5GB全国流量，流量有效期1个月':25,
    '20GB本地流量+3GB全国流量':23,
    '20G本地流量':20,
    '2GB本地流量':2,
    '2GB本地流量，流量有效期1个月':2,
    '2G本地流量':2,
    '300分钟+300MB':3.3,
    '300分钟本地拨打国内、1GB本地流量':4,
    '300分钟本地市话+300M本地流量':3.3,
    '3GB本地流量':3,
    '3G本地流量':3,
    '40BG本地流量':40,
    '40GB本地流量':40,
    '40GB本地流量，流量有效期1个月':40,
    '40G本地流量':40,
    '40G全国流量':40,
    '4G本地流量':4,
    '500M本地流量':0.5,
    '5GB本地流量':5,
    '5G本地流量':5,
    '600分钟+600MB':6.6,
    '600分钟本地拨打国内、2GB本地流量':8,
    '600分钟本地市话+600M本地流量':6.6,
    '60G本地流量':60,
    '6G本地流量':6,
    '800M本地流量':0.8,
    '8GB本地流量':8,
    '8G本地流量':8,
    '本地流量':80,
    '不限量':80
    }
data_cate['DX_FLUX']=data['DX_FLUX'].map(dx_flux_dic).fillna(0)
data_cate['DX_FLUX'].value_counts()
#data['DX_FLUX'].value_counts()
#len(data[data['DX_FLUX'].isnull()]) ###8887


# In[13]:



dummy_cols=['CITY_DESC','DESC_1ST','SYSTEM_FLAG','PRODUCT_TYPE','RH_TYPE','FLUX_RELEASE','SUB_PURCH_MODE_TYPE','PAYMENT']
frm_dummy=data_cate[dummy_cols]
frm_dummy=pd.get_dummies(frm_dummy)


# In[109]:


data['DX_NAME'].value_counts()


# In[16]:


#withoutSummer = odata.drop(['summer'],axis=1)
frm_dummy[['DX_FLUX','DX_EFF_DATE']]=data_cate[['DX_FLUX','DX_EFF_DATE']]
data_conti=data_conti.fillna(0)
data_conti['DX_HIGH_SPEED']=data_conti['DX_HIGH_SPEED'].map(lambda x: str(x).replace("G",""))
data_conti['OVER_VOICE_FEE_AVG']=data_conti['OVER_VOICE_FEE_AVG'].map(lambda x: str(x).replace("\"","").replace("," , ""))

frm_dummy[data_conti.columns.values]=data_conti
frm_dummy.head()
frm_final=frm_dummy
#for i in frm_final.columns.values:
#    if(frm_final[i].dtype=="str"):
#        frm_final[i]=pd.to_numeric(frm_final[i],errors='raise')
#    else:
#        continue


# In[17]:


#data_conti.describe()
print(data_cate.columns.values)
frm_dummy.columns.values


# In[18]:


from sklearn.decomposition import PCA
pca = PCA(n_components=0.99) #当选择mle的时候，第二个参数必须写。svd_solver='full'

'''
colss=[]
count=0
for i in frm_final.columns.values:
    print("\r\r\r~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(i)     
    if(count>0):
        colss.append(i)
        frm_test=frm_final[colss]
        #print(frm_test)
        pca.fit(frm_test)
    else:
        colss.append(i)
    count=count+1
'''
#pca.fit(frm_final)
pca.fit(frm_final[conti_cols])
variance_ratio_=pca.explained_variance_ratio_
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.n_components_)
#print(np.sum(variance_ratio_))

##https://blog.csdn.net/kancy110/article/details/73303125
# PCA(copy=True, n_components=2, whiten=False)
#rint(pca.explained_variance_ratio_)
#data['OVER_VOICE_FEE_AVG'].to_csv('data\\check_over_voice_fee_avg.csv',index=False)


# In[168]:


pca.components_


# In[22]:


from numpy import *

#解析文本数据函数
#@filename 文件名txt
#@delim 每一行不同特征数据之间的分隔方式，默认是tab键'\t'
def loadDataSet(filename,delim='\t'):
    #打开文本文件
    fr=open(filename)
    #对文本中每一行的特征分隔开来，存入列表中，作为列表的某一行
    #行中的每一列对应各个分隔开的特征
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    #利用map()函数，将列表中每一行的数据值映射为float型
    datArr=[map(float.line)for line in stringArr]
    #将float型数据值的列表转化为矩阵返回
    return mat(datArr)

#pca特征维度压缩函数
#@dataMat 数据集矩阵
#@topNfeat 需要保留的特征维度，即要压缩成的维度数，默认4096    
def pca(dataMat,topNfeat=4096):
    #求数据矩阵每一列的均值
    meanVals=mean(dataMat,axis=0)
    #数据矩阵每一列特征减去该列的特征均值
    meanRemoved=dataMat-meanVals
    #计算协方差矩阵，除数n-1是为了得到协方差的无偏估计
    #cov(X,0) = cov(X) 除数是n-1(n为样本个数)
    #cov(X,1) 除数是n
    covMat=cov(meanRemoved,rowvar=0)
    #计算协方差矩阵的特征值及对应的特征向量
    #均保存在相应的矩阵中
    eigVals,eigVects=linalg.eig(mat(covMat))
    #sort():对特征值矩阵排序(由小到大)
    #argsort():对特征值矩阵进行由小到大排序，返回对应排序后的索引
    eigValInd=argsort(eigVals)
    #从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    #将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    redEigVects=eigVects[:,eigValInd]
    #将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat=meanRemoved*redEigVects
    #利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    reconMat=(lowDDataMat*redEigVects.T)+meanVals
    #返回压缩后的数据矩阵即该矩阵反构出原始数据矩阵
    return lowDDataMat,reconMat
for i in frm_final.columns.values:
    if(frm_final[i].dtype=="str"):
        frm_final[i]=pd.to_numeric(frm_final[i],errors='raise')
    else:
        continue


dataMat=frm_final[data_conti.columns.values].as_matrix()
dataMat = dataMat.astype(float64)

meanVals=mean(dataMat,axis=0)
meanRemoved=dataMat-meanVals
covMat=cov(meanRemoved,rowvar=0)
eigVals,eigVects=linalg.eig(mat(covMat))
eigValInd=argsort(eigVals)
eigValInd


# In[26]:


conti_cols_final=data_conti.columns.values
all_cols_final=frm_final.columns.values


# In[27]:


col_names=np.array(frm_final.columns.values)
feature_dic={}
for i,arr_inx in zip(eigValInd,range(94)):
    feature_dic[arr_inx+1]=conti_cols_final[i]

feature_dic  


# In[28]:


dataMat=frm_final.as_matrix()
dataMat = dataMat.astype(float64)

meanVals=mean(dataMat,axis=0)
meanRemoved=dataMat-meanVals
covMat=cov(meanRemoved,rowvar=0)
eigVals,eigVects=linalg.eig(mat(covMat))
eigValInd=argsort(eigVals)
eigValInd


# In[29]:


all_cols_final

#col_names=np.array(frm_final.columns.values)
feature_dic={}
for i,arr_inx in zip(eigValInd,range(len(all_cols_final))):
    feature_dic[arr_inx+1]=all_cols_final[i]

feature_dic  

