
# coding: utf-8

# In[86]:


import pandas as pd
df_ori = pd.read_csv('data\\dw_cl_jl_high_value.csv',delimiter='|')
df_user = pd.read_csv('data\\all_user_7in8not.csv')
data=pd.merge(df_user,df_ori,how='left',left_on='USER_ID_7in8not',right_on='USER_ID')
data.head()


# In[87]:


category_cols=['CITY_DESC','DESC_1ST','SYSTEM_FLAG','PRODUCT_TYPE','RH_TYPE',
           'FLUX_RELEASE','DX_FLUX','DX_EFF_DATE','SUB_PURCH_MODE_TYPE','PAYMENT']
data_cate=data[category_cols]


# ##DX_EFF_DATE

# In[88]:


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
#


# In[89]:


data_cate['CITY_DESC']=data['CITY_DESC'].fillna('未知').astype('str')
data_cate['CITY_DESC']=data['CITY_DESC'].map(lambda x:str(x).replace('主城未知','未知').replace('nan','未知'))
data_cate['CITY_DESC'].value_counts()
#data_cate['DX_EFF_DATE'].value_counts()


# In[90]:


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


# In[91]:


data_cate['RH_TYPE']=data['RH_TYPE'].fillna('非融合').astype('str')
print(data_cate['RH_TYPE'].value_counts())
data['SYSTEM_FLAG'].value_counts()


# In[92]:


data_cate['SUB_PURCH_MODE_TYPE'] =data_cate['SUB_PURCH_MODE_TYPE'].fillna('其他')
data_cate['SUB_PURCH_MODE_TYPE'].value_counts()
data_cate['PAYMENT'] =data_cate['PAYMENT'].fillna('其他')
data_cate['PAYMENT'].value_counts()


# In[93]:


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


# In[94]:


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


# In[95]:


dummy_cols=['CITY_DESC','DESC_1ST','SYSTEM_FLAG','PRODUCT_TYPE','RH_TYPE','FLUX_RELEASE','SUB_PURCH_MODE_TYPE','PAYMENT']
frm_dummy=data_cate[dummy_cols]
frm_dummy=pd.get_dummies(frm_dummy)


# In[96]:


scatter_cols=frm_dummy.columns.values.tolist()
ori_binary_cols=['HAS_FK','HAS_ADSL','WXBD','STBD','OPPOSE_LINK','IS_YH','IS_TERM','IS_FP_PRINT',
                 'IS_TS','IS_LH','IS_VIDEO','IS_CHANGE','RELEASE_FLAG','IS_ZK','YY_FLAG']
conti_cols=['INNET_DATE','PRODUCT_FEE'  #,'TOTAL_FLUX_THIS','FEE_THIS',,'CALLING_DURA_THIS','CHARGE_FLUX_08','CHARGE_VOICE_08','INNER_ROAM_FEE_08'
,'PROD_IN_VOICE','PROD_IN_FLUX','CALL_NUM','TERM_END_MONTH','LH_END_MONTH'
,'DX_FEE','DX_END_DATE','DX_HIGH_SPEED','ARPU_AVG','FEE_LAST1','FEE_LAST2','FEE_LAST3'
,'FEE_LAST4','FEE_LAST5','CALLING_DURA_AVG_3','CALLING_DURA_LAST1','CALLING_DURA_LAST2','CALLING_DURA_LAST3'
,'CALLING_DURA_LAST4','CALLING_DURA_LAST5','AVG_FLUX_3','TOTAL_FLUX_LAST1','TOTAL_FLUX_LAST2','TOTAL_FLUX_LAST3'
,'TOTAL_FLUX_LAST4','TOTAL_FLUX_LAST5','EX_FLUX_FEE_LAST1','EX_FLUX_FEE_LAST2','EX_FLUX_FEE_LAST3' #,'EX_FLUX_FEE_THIS'
,'EX_FLUX_FEE_LAST4','EX_FLUX_FEE_LAST5','PKG_FLUX_FEE_LAST1','PKG_FLUX_FEE_LAST2','PKG_FLUX_FEE_LAST3' #'PKG_FLUX_FEE_THIS',
,'PKG_FLUX_FEE_LAST4','PKG_FLUX_FEE_LAST5','VOICE_FEE_LAST1','VOICE_FEE_LAST2','VOICE_FEE_LAST3','VOICE_FEE_LAST4' #'VOICE_FEE_THIS',
,'VOICE_FEE_LAST5','CHARGE_FLUX_07','CHARGE_VOICE_07','CDR_KF_OUT','CDR_KF_IN','YH_NUM','INNER_ROAM_FEE_07','INNER_ROAM_FEE_06','PRINT_CNT'
,'TS_CNT','OVER_FLUX_FEE_AVG','BUILDING_INFO','OVER_VOICE_FEE_AVG','DX_HIGH_SPEED','DX_FLUX','DX_EFF_DATE']
#for i in ori_binary_cols:
#    scatter_cols.append( i)
    
frm_scatter=frm_dummy
frm_scatter[ori_binary_cols]=data[ori_binary_cols]
frm_scatter=frm_scatter.fillna(0)

frm_conti=data[conti_cols]
frm_conti=frm_conti.fillna(0)
frm_conti['DX_HIGH_SPEED']=frm_conti['DX_HIGH_SPEED'].applymap(lambda x: str(x).replace("G",""))
frm_conti['OVER_VOICE_FEE_AVG']=frm_conti['OVER_VOICE_FEE_AVG'].map(lambda x: str(x).replace("\"","").replace("," , ""))
frm_conti[['DX_FLUX','DX_EFF_DATE']]=data_cate[['DX_FLUX','DX_EFF_DATE']]
frm_conti.head()


# In[116]:


frm_conti['FEE_LAST1-2']=(frm_conti['FEE_LAST1']-frm_conti['FEE_LAST2']) / ( frm_conti['FEE_LAST2'] + 1)
frm_conti['FEE_LAST2-3']=(frm_conti['FEE_LAST2']-frm_conti['FEE_LAST3']) / (frm_conti['FEE_LAST3'] + 1 )
frm_conti['FEE_LAST3-4']=(frm_conti['FEE_LAST3']-frm_conti['FEE_LAST4']) / (frm_conti['FEE_LAST4'] + 1 )
frm_conti['FEE_LAST4-5']=(frm_conti['FEE_LAST4']-frm_conti['FEE_LAST5']) / (frm_conti['FEE_LAST5'] + 1 )

frm_conti['CALLING_DURA_LAST1-2']=(frm_conti['CALLING_DURA_LAST1']-frm_conti['CALLING_DURA_LAST2'])/(frm_conti['CALLING_DURA_LAST2']+1)
frm_conti['CALLING_DURA_LAST2-3']=(frm_conti['CALLING_DURA_LAST2']-frm_conti['CALLING_DURA_LAST3'])/(frm_conti['CALLING_DURA_LAST3']+1)
frm_conti['CALLING_DURA_LAST3-4']=(frm_conti['CALLING_DURA_LAST3']-frm_conti['CALLING_DURA_LAST4'])/(frm_conti['CALLING_DURA_LAST4']+1)
frm_conti['CALLING_DURA_LAST4-5']=(frm_conti['CALLING_DURA_LAST4']-frm_conti['CALLING_DURA_LAST5'])/(frm_conti['CALLING_DURA_LAST5']+1)

frm_conti['TOTAL_FLUX_LAST1-2']=(frm_conti['TOTAL_FLUX_LAST1']-frm_conti['TOTAL_FLUX_LAST2'])/(frm_conti['TOTAL_FLUX_LAST2']+1)
frm_conti['TOTAL_FLUX_LAST2-3']=(frm_conti['TOTAL_FLUX_LAST2']-frm_conti['TOTAL_FLUX_LAST3'])/(frm_conti['TOTAL_FLUX_LAST3']+1)
frm_conti['TOTAL_FLUX_LAST3-4']=(frm_conti['TOTAL_FLUX_LAST3']-frm_conti['TOTAL_FLUX_LAST4'])/(frm_conti['TOTAL_FLUX_LAST4']+1)
frm_conti['TOTAL_FLUX_LAST4-5']=(frm_conti['TOTAL_FLUX_LAST4']-frm_conti['TOTAL_FLUX_LAST5'])/(frm_conti['TOTAL_FLUX_LAST5']+1)

frm_conti['EX_FLUX_FEE_LAST1-2']=(frm_conti['EX_FLUX_FEE_LAST1']-frm_conti['EX_FLUX_FEE_LAST2'])/(frm_conti['EX_FLUX_FEE_LAST2']+1)
frm_conti['EX_FLUX_FEE_LAST2-3']=(frm_conti['EX_FLUX_FEE_LAST2']-frm_conti['EX_FLUX_FEE_LAST3'])/(frm_conti['EX_FLUX_FEE_LAST3']+1)
frm_conti['EX_FLUX_FEE_LAST3-4']=(frm_conti['EX_FLUX_FEE_LAST3']-frm_conti['EX_FLUX_FEE_LAST4'])/(frm_conti['EX_FLUX_FEE_LAST4']+1)
frm_conti['EX_FLUX_FEE_LAST4-5']=(frm_conti['EX_FLUX_FEE_LAST4']-frm_conti['EX_FLUX_FEE_LAST5'])/(frm_conti['EX_FLUX_FEE_LAST5']+1)

frm_conti['PKG_FLUX_FEE_LAST1-2']=(frm_conti['PKG_FLUX_FEE_LAST1']-frm_conti['PKG_FLUX_FEE_LAST2'])/(frm_conti['PKG_FLUX_FEE_LAST2']+1)
frm_conti['PKG_FLUX_FEE_LAST2-3']=(frm_conti['PKG_FLUX_FEE_LAST2']-frm_conti['PKG_FLUX_FEE_LAST3'])/(frm_conti['PKG_FLUX_FEE_LAST3']+1)
frm_conti['PKG_FLUX_FEE_LAST3-4']=(frm_conti['PKG_FLUX_FEE_LAST3']-frm_conti['PKG_FLUX_FEE_LAST4'])/(frm_conti['PKG_FLUX_FEE_LAST4']+1)
frm_conti['PKG_FLUX_FEE_LAST4-5']=(frm_conti['PKG_FLUX_FEE_LAST4']-frm_conti['PKG_FLUX_FEE_LAST5'])/(frm_conti['PKG_FLUX_FEE_LAST5']+1)

frm_conti['VOICE_FEE_LAST1-2']=(frm_conti['VOICE_FEE_LAST1']-frm_conti['VOICE_FEE_LAST2'])/(frm_conti['VOICE_FEE_LAST2']+1)
frm_conti['VOICE_FEE_LAST2-3']=(frm_conti['VOICE_FEE_LAST2']-frm_conti['VOICE_FEE_LAST3'])/(frm_conti['VOICE_FEE_LAST3']+1)
frm_conti['VOICE_FEE_LAST3-4']=(frm_conti['VOICE_FEE_LAST3']-frm_conti['VOICE_FEE_LAST4'])/(frm_conti['VOICE_FEE_LAST4']+1)
frm_conti['VOICE_FEE_LAST4-5']=(frm_conti['VOICE_FEE_LAST4']-frm_conti['VOICE_FEE_LAST5'])/(frm_conti['VOICE_FEE_LAST5']+1)


# In[117]:


from sklearn.preprocessing import MinMaxScaler
#scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
frm_cont_out = pd.DataFrame(min_max_scaler.fit_transform(frm_conti),columns=frm_conti.columns.values)
frm_final=frm_cont_out#_out
frm_final[frm_scatter.columns.values]=frm_scatter

frm_final.describe()


# In[128]:


from  sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
selector=SelectKBest(chi2,k=120)
#
frm_final.index=data['USER_ID_7in8not']
selector.fit(frm_final,data['lost_tag'])
feature_matrix=selector.get_support()
#pvalues=selector.scores_ 
#print(pvalues)
#pvalues
#Frm_New=selector.fit_transform(frm_final,data['lost_tag'].values)
#feature_matrix.shape
col_names=np.array(frm_final.columns.values)
feature_survice=[]
#i=0
for bool_value,arr_inx in zip(feature_matrix,range(198)):
    if bool_value==True:
        #print("count:"+str(arr_inx)+"  ,bool value is"+str(bool_value))
        feature_survice.append(col_names[arr_inx])
        #[i+1]=col_names[arr_inx]
        #i=i+1
    
feature_survice


# # 以下是互信息法
# 

# In[119]:


from  sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

selector=SelectKBest(mutual_info_classif,k=50)
#
frm_final.index=data['USER_ID_7in8not']
selector.fit(frm_final,data['lost_tag'])
feature_matrix=selector.get_support()
pvalues=selector.scores_ 
#print(pvalues)
#pvalues
#Frm_New=selector.fit_transform(frm_final,data['lost_tag'].values)
#feature_matrix.shape
col_names=np.array(frm_final.columns.values)
feature_survice=[]
#i=0
for bool_value,arr_inx in zip(feature_matrix,range(198)):
    if bool_value==True:
        #print("count:"+str(arr_inx)+"  ,bool value is"+str(bool_value))
        feature_survice.append(col_names[arr_inx])
        #[i+1]=col_names[arr_inx]
        #i=i+1   
feature_survice


# In[120]:


import numpy as np
col_indx=np.argsort(pvalues)
#sort_feature_frm=pd.DataFrame()
feature_dic={}
for i,arr_inx in zip(col_indx,range(174)):
    feature_dic[int(arr_inx)+1]=col_names[i]
feature_dic  


# # 一下是SelectKBest的 f_classif

# In[121]:


from  sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

selector=SelectKBest(f_classif,k=50)
#
frm_final.index=data['USER_ID_7in8not']
selector.fit(frm_final,data['lost_tag'])
feature_matrix=selector.get_support()
pvalues_1=selector.scores_ 
#print(pvalues)
#pvalues
#Frm_New=selector.fit_transform(frm_final,data['lost_tag'].values)
#feature_matrix.shape
col_names=np.array(frm_final.columns.values)
feature_survice=[]
#i=0
for bool_value,arr_inx in zip(feature_matrix,range(198)):
    if bool_value==True:
        #print("count:"+str(arr_inx)+"  ,bool value is"+str(bool_value))
        feature_survice.append(col_names[arr_inx])
        #[i+1]=col_names[arr_inx]
        #i=i+1   
feature_survice


# In[123]:


import numpy as np
col_indx_1=np.argsort(pvalues_1)
#sort_feature_frm=pd.DataFrame()
feature_dic={}
for i,arr_inx in zip(col_indx_1,range(198)):
    feature_dic[int(arr_inx)+1]=col_names[i]
feature_dic  


# # 基于树模型的特征选择和SVM（有点像跑模型了）

# In[108]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
#linear kernel
#data['lost_tag'].value_counts()
clf.fit(frm_final,data['lost_tag'])
clf.coef_


# In[124]:



#GBDT作为基模型的特征选择
clf_gbdt=GradientBoostingClassifier()
clf_gbdt.fit(frm_final,data['lost_tag'])
bool_inx=clf_gbdt.feature_importances_
bool_inx


# In[125]:


import numpy as np
col_indx=np.argsort(bool_inx)
#sort_feature_frm=pd.DataFrame()
feature_dic={}
for i,arr_inx in zip(col_indx,range(198)):
    feature_dic[int(arr_inx)+1]=col_names[i]
feature_dic  


# # 以下是Variance Threadshold方法选取离散特征。

# In[21]:


print(data_cate.columns.values)
data.columns.values


# In[30]:


train_data_cate=frm_dummy[['CITY_DESC_万州', 'CITY_DESC_两江新区', 'CITY_DESC_丰都', 'CITY_DESC_九龙坡',
       'CITY_DESC_云阳', 'CITY_DESC_北碚', 'CITY_DESC_南岸', 'CITY_DESC_南川',
       'CITY_DESC_合川', 'CITY_DESC_垫江', 'CITY_DESC_城口', 'CITY_DESC_大渡口',
       'CITY_DESC_大足', 'CITY_DESC_奉节', 'CITY_DESC_巫山', 'CITY_DESC_巫溪',
       'CITY_DESC_巴南', 'CITY_DESC_开州区', 'CITY_DESC_彭水', 'CITY_DESC_忠县',
       'CITY_DESC_未知', 'CITY_DESC_梁平', 'CITY_DESC_武隆', 'CITY_DESC_永川',
       'CITY_DESC_江北', 'CITY_DESC_江津', 'CITY_DESC_沙坪坝', 'CITY_DESC_涪陵城区',
       'CITY_DESC_渝中', 'CITY_DESC_渝北', 'CITY_DESC_潼南', 'CITY_DESC_璧山',
       'CITY_DESC_电子商务部', 'CITY_DESC_石柱', 'CITY_DESC_秀山', 'CITY_DESC_綦江',
       'CITY_DESC_荣昌', 'CITY_DESC_酉阳', 'CITY_DESC_铜梁', 'CITY_DESC_长寿',
       'CITY_DESC_集团客户部', 'CITY_DESC_黔江主城区', 'DESC_1ST_OCS充值',
       'DESC_1ST_OCS有效', 'DESC_1ST_停机', 'DESC_1ST_拆机', 'DESC_1ST_挂失',
       'DESC_1ST_正常在用', 'DESC_1ST_限制呼出', 'SYSTEM_FLAG_OCS-BSS',
       'SYSTEM_FLAG_非OCS-BSS', 'PRODUCT_TYPE_全家享', 'PRODUCT_TYPE_冰激凌',
       'PRODUCT_TYPE_智慧沃家', 'PRODUCT_TYPE_标准资费', 'PRODUCT_TYPE_非标准资费',
       'RH_TYPE_BSS全家享主卡', 'RH_TYPE_BSS全家享副卡', 'RH_TYPE_BSS其他融合产品',
       'RH_TYPE_当前证件下有宽带', 'RH_TYPE_智慧沃家共享版', 'RH_TYPE_非融合',
       'FLUX_RELEASE_全家享主卡', 'FLUX_RELEASE_全家享副卡', 'FLUX_RELEASE_冰激凌单卡',
       'FLUX_RELEASE_政企低消', 'FLUX_RELEASE_普通低消', 'FLUX_RELEASE_未释放',
       'FLUX_RELEASE_畅越低消', 'FLUX_RELEASE_订购不限量包', 'FLUX_RELEASE_预约冰激凌',
       'SUB_PURCH_MODE_TYPE_低消', 'SUB_PURCH_MODE_TYPE_低消承诺送流量',
       'SUB_PURCH_MODE_TYPE_其他', 'SUB_PURCH_MODE_TYPE_承诺话费送语音',
       'SUB_PURCH_MODE_TYPE_政企低消', 'SUB_PURCH_MODE_TYPE_畅爽',
       'SUB_PURCH_MODE_TYPE_畅越低消', 'PAYMENT_MISPOS(界面控制)',
       'PAYMENT_MISPOS刷卡缴费', 'PAYMENT_其他', 'PAYMENT_合约返费/赠费',
       'PAYMENT_存款清退_普通预存款', 'PAYMENT_总部电子渠道支付宝缴费', 'PAYMENT_挂账缴费',
       'PAYMENT_月结协议预存款销帐', 'PAYMENT_月结普通预存款销帐', 'PAYMENT_沃支付银行卡代扣',
       'PAYMENT_现金交费', 'PAYMENT_缴费卡实时销帐', 'PAYMENT_缴费卡收入_普通预存款',
       'PAYMENT_自助终端现金', 'PAYMENT_营业厅收入(帐务收费)_普通预存款', 'PAYMENT_资金归集现金缴费',
       'PAYMENT_转帐（转出）', 'PAYMENT_退预存款', 'PAYMENT_银行代收_普通预存款',
       'PAYMENT_银行联网交费','IS_LH','HAS_FK', 'HAS_ADSL', 'IS_FP_PRINT',
        'IS_TS',  'IS_VIDEO', 'IS_CHANGE', 'IS_GWLS',
       'RELEASE_FLAG',  'YY_FLAG', 'BUILDING_INFO', 'IS_ZK']]
#train_data_cate=data['lost_tag']
frm_dummy.columns.values


# In[68]:


#VarianceThreshold 处理离散型变量，也就是哑变量和
from sklearn.feature_selection import VarianceThreshold 
sel=VarianceThreshold(threshold=(0.8*(1-0.8)))#表示剔除特征的方差大于阈值的特征Removing features with low variance
sel.fit_transform(train_data_cate)#返回的结果为选择的特征矩阵
print(sel.fit_transform(train_data_cate))
bool_inx=sel.get_support()
survive_cols=[]
all_cols=train_data_cate.columns.values
for i in range(len(bool_inx)):
    if(bool_inx[i]==True):
        survive_cols.append(all_cols[i])
survive_cols


# # 以下是矩阵计算的过程

# In[35]:


matrixa=np.array([[2,5,2],[3,8,2],[7,6,3],
                  [2,8,4],[8,8,3],[6,4,5]])
a=np.cov(matrixa,rowvar=0)
a


# In[41]:


eigVals,eigVects=np.linalg.eig(np.mat(a))
eigVals


# In[42]:


eigVects


# In[46]:



def eigValPct(eigVals,percentage):
    sortArray=np.sort(eigVals) #使用numpy中的sort()对特征值按照从小到大排序
    sortArray=sortArray[-1::-1] #特征值从大到小排序
    arraySum=np.sum(sortArray) #数据全部的方差arraySum
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if tempSum>=arraySum*percentage:
            return num
eigValInd=np.argsort(eigVals)
eigValInd=eigValInd[:-(2+1):-1] #要两个特征
eigValInd


# In[47]:


redEigVects=eigVects[:,eigValInd]
redEigVects

