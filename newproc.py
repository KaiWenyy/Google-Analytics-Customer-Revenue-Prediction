import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import NaN

import json
from pandas.io.json import json_normalize
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')


JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource'] # no 'hits' 'customDimensions'
NEW_COLUMNS = ['customDimensions'] #no hits

SIMPLE_COLUMNS = ['channelGrouping', 'date', 'socialEngagementType','visitId', 'visitNumber', 'visitStartTime','fullVisitorId']

#============================================
# load function
# !!!!!need to notice the "dtype" argument 
#============================================
def load_other_df(filename):
    print("load file " + filename + '...')
    df = pd.read_csv(filename,
        dtype = {'fullVisitorId': 'str', 'visitId': 'str', 
                "totals.transactionRevenue": 'float'}
        )
    return df
def load_hits_df(filename):
    print("load file " + filename + '...')
    df = pd.read_csv(filename,
        dtype = {'ind': 'float'}
        )
    return df

#============================================
# for hits df 
#============================================
def load_df2(csv_name, nrows = None, isTrain = True, selectID = None, numFile =0):
    "csv_path：檔案路徑， nrows 讀取行數，JSON_COLUMNS: JSON的列"

    df = pd.read_csv(csv_name,
                     converters = {column: json.loads for column in JSON_COLUMNS},
                     # json.loads : json --> python
                     dtype = {'fullVisitorId': 'str', 'visitId': 'str'},
                     nrows = nrows)

    for column in JSON_COLUMNS + NEW_COLUMNS + SIMPLE_COLUMNS:
        df.drop(column, axis = 1, inplace = True)

    if isTrain == False:
        '''
        repeatId, n,m = get_repeatID()
        for i in df.index:
            if df['fullVisitorId'][i] not in repeatId:
                df.drop([index], inplace = True)
        df.reset_index()
        '''
        s = [i - (numFile*5000) for i in selectID if i >= (numFile*5000) and i < ((numFile+1)*5000)]
        print('selected rows: ', len(s))
        df = df.iloc[s]


    col = 'hits'
    df[col][df[col] == "[]"] = "[{}]"
    df[col] = df[col].apply(literal_eval)
    #print(type(df[col][0]), len(df[col][0]))
    #print(type(df[col][0][0]), len(df[col][0][0]))
    
    ind = 0  
    hit_df = pd.DataFrame(columns = ['hits','ind'])
    for sample in df[col]:
        if ind in 1000*(1,2,3,4):
            print(ind)
        for h in sample:
            insert = pd.DataFrame([[h,ind]], columns=['hits','ind'])
            #print(insert)
            hit_df = hit_df.append(insert,ignore_index=True)
        ind += 1

    #d['hits'] = d['hits'].apply(literal_eval)
    column_as_df = json_normalize(hit_df['hits'])
    column_as_df.columns = [f"{'hits'}.{subcol}" for subcol in column_as_df.columns]

    # Extract the product and promo names from the complex nested structure into a simple flat list:
    if 'hits.product' in column_as_df.columns:
        column_as_df['hits.v2ProductName'] = column_as_df['hits.product'].apply(lambda x: [p['v2ProductName'] for p in x] if type(x) == list else [])
        column_as_df['hits.v2ProductCategory'] = column_as_df['hits.product'].apply(lambda x: [p['v2ProductCategory'] for p in x] if type(x) == list else [])
        del column_as_df['hits.product']
        
    if 'hits.promotion' in column_as_df.columns:
        column_as_df['hits.promoId'] = column_as_df['hits.promotion'].apply(lambda x: [p['promoId'] for p in x] if type(x) == list else [])
        column_as_df['hits.promoName'] = column_as_df['hits.promotion'].apply(lambda x: [p['promoName'] for p in x] if type(x) == list else [])
        del column_as_df['hits.promotion']

    hit_df = hit_df.drop(col, axis = 1).merge(column_as_df, left_index = True, right_index = True) 
    
    # drop not available
    drop_col = ['hits.experiment', 'hits.customVariables', 
                'hits.customDimensions', 'hits.customDimensions',
                'hits.publisher_infos', 'hits.appInfo.screenDepth',
                'hits.appInfo.screenDepth','hits.contentGroup.contentGroup5',
                'hits.latencyTracking.pageLoadSample', 'hits.latencyTracking.pageLoadTime',
                'hits.latencyTracking.pageDownloadTime', 'hits.latencyTracking.redirectionTime',
                'hits.latencyTracking.redirectionTime', 'hits.latencyTracking.redirectionTime',
                'hits.latencyTracking.redirectionTime', 'hits.latencyTracking.serverResponseTime',
                'hits.latencyTracking.domLatencyMetricsSample','hits.latencyTracking.domInteractiveTime',
                'hits.latencyTracking.domContentLoadedTime','hits.latencyTracking.speedMetricsSample',
                'hits.latencyTracking.domainLookupTime','hits.latencyTracking.serverConnectionTime',
                'hits.customMetrics', 'hits.social.socialInteractionNetworkAction',
                'hits.page.searchKeyword', 'hits.page.searchCategory',
                ]
    for c in drop_col:
        if c in hit_df.columns:
            #print("drop "+ c)
            #del hit_df[c]
            hit_df.drop(c, axis = 1, inplace = True)

    #for c in hit_df.columns:
    #    print(c, hit_df[c].count())

    #print(hit_df.columns)
    #print(len(hit_df.columns))
    hit_df.to_csv('hit_exjson_' + csv_name.split('.')[0] + '.csv', index = False)
    return hit_df, len(s)

#============================================
# for others df
#============================================
def load_df(csv_name, nrows = None):
    "csv_path：檔案路徑， nrows 讀取行數，JSON_COLUMNS: JSON的列"

    df = pd.read_csv(csv_name,
                     converters = {column: json.loads for column in JSON_COLUMNS},
                     # json.loads : json --> python
                     dtype = {'fullVisitorId': 'str', 'visitId': 'str'},
                     nrows = nrows)
    
    #drop hits columns
    df = df.drop('hits', axis = 1)

    for col in NEW_COLUMNS:
        df[col][df[col] == "[]"] = "[{}]"
        df[col] = df[col].apply(literal_eval).str[0]
        
    for column in JSON_COLUMNS + NEW_COLUMNS:
        column_as_df = json_normalize(df[column])
        # json column --> tabel(DataFrame)
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        # f-string in Python 3.6
        df = df.drop(column, axis = 1).merge(column_as_df, left_index = True, right_index = True)
        
    df.to_csv('exjson_' + csv_name.split('.')[0] + '.csv', index = False)        
    return df

def exjson(path, num, isTrain= True, selectID = None):   
    #os.chdir(path)
    files = [ os.path.join(path, 'test_split') + str(d+242) + '.csv' for d in range(num)]

    n = 0
    t_len = 0
    for i in files:
        #load_df(i)
        tmp, s = load_df2(i, isTrain = isTrain, selectID = selectID, numFile =n )
        print('No. {} is done.'.format(i.split('.')[0]))
        n+=1
        t_len += s
    assert t_len == len(selectID)

#============================================
# concat functions
#============================================
def concat_df(path, num, outname = None):
    "path: path_train/path_test; num: 86/21"
    os.chdir(path)
    file_list = ['test_split{}.csv'.format(i+1) for i in range(num)]
    df_list = []
    
    for file in file_list:
        dfname = file.split('.')[0]
        dfname = pd.read_csv(file, dtype = {'fullVisitorId': 'str', 'visitId': 'str'})
        df_list.append(dfname)
        
    df = pd.concat(df_list, ignore_index = True)
    df.to_csv(outname, index = False)
    return df

def concat_df_hit(path, num, outname = None):
    "path: path_train/path_test; num: 86/21"
    os.chdir(path)
    file_list = ['train_split{}.csv'.format(i+242) for i in range(num)]
    df_list = []
    
    pre_len = 0
    n = 0
    for file in file_list:
        dfname = file.split('.')[0]
        dfname = pd.read_csv(file, dtype = {'ind': 'float'})
        print(n, dfname.iloc[-1, 0]+1, len(dfname.index), pre_len)
        dfname['ind'] = dfname['ind'] + pre_len
        df_list.append(dfname)
        pre_len = dfname.iloc[-1, 0]+1
        n+=1
        
    df = pd.concat(df_list, ignore_index = True)
    df.to_csv(outname, index = False)
    return df

#============================================
# select test row 
# with the same ID in train data
#============================================
def get_repeatID():
    train = load_dataFrame('exjson_train_split2/train_df.csv')
    test = load_dataFrame('exjson_test_split/test_df.csv')
    repeatId = intersection(train['fullVisitorId'].unique(), test['fullVisitorId'].unique())

    return repeatId, train, test

def get_test_no_hit():
    repeatId, train, test = get_repeatID()
    selectID = []
    col = 'fullVisitorId'
    selectID = []
    df_list = []
    n = 1
    for i in repeatId :
        print(n,'/',len(repeatId))
        ind_list = test[col][ test[col] == i].index
        #print(list(ind))
        selectID.extend(list(ind_list))
        n+= 1
    selectID = sorted(selectID)
    test = test.iloc[selectID]
    df_list.append(test)
    test = pd.concat(df_list, ignore_index = True)
    test.to_csv('exjson_test_split/test_selecet_df.csv', index = False)

    return test, selectID


#============================================
# utils
#============================================
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def unique(list1): 
    x = np.array(list1) 
    print('u', len(np.unique(x)))
    return np.unique(x)

def bug_fix(df):
    drop_list = df[df['date'] == "No"].index.tolist()
    df = df.drop(drop_list)
    return df

def bug_hit_fix(df):
    nan = df['ind'][ df['ind'].isnull()].index
    df.drop( nan, inplace = True)
    return df

def main():
    # train hit
    exjson('train_split2', 1, isTrain = True)
    df = concat_df_hit('hit_exjson_train_split2', 101, 'train_df.csv')
    #print(df['ind'][12823:12835])