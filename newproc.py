import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
from pandas.io.json import json_normalize
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')


JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource'] # no 'hits' 'customDimensions'
NEW_COLUMNS = ['customDimensions'] #no hits

def split_df(df, path, num_split):

    os.chdir(path)            
    
    for i in range(num_split):
        temp = df[i*20000 : (i+1)*20000]
        temp.to_csv(str(i) + '.csv', index = False)
        print('No. %s is done.' %i)


def load_df2(csv_name, nrows = None):
    "csv_path：檔案路徑， nrows 讀取行數，JSON_COLUMNS: JSON的列"

    df = pd.read_csv(csv_name,
                     converters = {column: json.loads for column in JSON_COLUMNS},
                     # json.loads : json --> python
                     dtype = {'fullVisitorId': 'str', 'visitId': 'str'},
                     nrows = nrows)


    col = 'hits'
    df[col][df[col] == "[]"] = "[{}]"
    df[col] = df[col].apply(literal_eval).str
    #print(len(df[col].apply(literal_eval).str))
    #print(df[col][32])
    
    column_as_df = json_normalize(df[column]) 
    column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
    df = df.drop(column, axis = 1).merge(column_as_df, left_index = True, right_index = True) 
        
    for column in JSON_COLUMNS + NEW_COLUMNS:
        df = df.drop(column, axis = 1)
        
       
        '''
        # Extract the product and promo names from the complex nested structure into a simple flat list:
        if 'hits.product' in column_as_df.columns:
            column_as_df['hits.v2ProductName'] = column_as_df['hits.product'].apply(lambda x: [p['v2ProductName'] for p in x] if type(x) == list else [])
            column_as_df['hits.v2ProductCategory'] = column_as_df['hits.product'].apply(lambda x: [p['v2ProductCategory'] for p in x] if type(x) == list else [])
            del column_as_df['hits.product']
            
        if 'hits.promotion' in column_as_df.columns:
            column_as_df['hits.promoId'] = column_as_df['hits.promotion'].apply(lambda x: [p['promoId'] for p in x] if type(x) == list else [])
            column_as_df['hits.promoName'] = column_as_df['hits.promotion'].apply(lambda x: [p['promoName'] for p in x] if type(x) == list else [])
            del column_as_df['hits.promotion']
        '''
            
        
        
    df.to_csv('exjson_' + csv_name.split('.')[0] + '.csv', index = False)        
    return df


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
        #print(len(df[col].apply(literal_eval).str))
        #print(df[col][32])
        
        
    for column in JSON_COLUMNS + NEW_COLUMNS:
        column_as_df = json_normalize(df[column])
        # json column --> tabel(DataFrame)
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        # f-string in Python 3.6
        
        '''
        # Extract the product and promo names from the complex nested structure into a simple flat list:
        if 'hits.product' in column_as_df.columns:
            column_as_df['hits.v2ProductName'] = column_as_df['hits.product'].apply(lambda x: [p['v2ProductName'] for p in x] if type(x) == list else [])
            column_as_df['hits.v2ProductCategory'] = column_as_df['hits.product'].apply(lambda x: [p['v2ProductCategory'] for p in x] if type(x) == list else [])
            del column_as_df['hits.product']
            
        if 'hits.promotion' in column_as_df.columns:
            column_as_df['hits.promoId'] = column_as_df['hits.promotion'].apply(lambda x: [p['promoId'] for p in x] if type(x) == list else [])
            column_as_df['hits.promoName'] = column_as_df['hits.promotion'].apply(lambda x: [p['promoName'] for p in x] if type(x) == list else [])
            del column_as_df['hits.promotion']
        '''
            
        df = df.drop(column, axis = 1).merge(column_as_df, left_index = True, right_index = True)
        
    df.to_csv('exjson_' + csv_name.split('.')[0] + '.csv', index = False)        
    return df

def exjson(path, num):
    
    #os.chdir(path)
    files = [ os.path.join(path, 'train_split') + str(d+1) + '.csv' for d in range(num)]

    for i in files:
        load_df(i)
        print('No. {} is done.'.format(i.split('.')[0]))

def concat_df(path, num, outname = None):
    "path: path_train/path_test; num: 86/21"
    os.chdir(path)
    file_list = ['train_split{}.csv'.format(i+1) for i in range(num)]
    df_list = []
    
    for file in file_list:
        dfname = file.split('.')[0]
        dfname = pd.read_csv(file, dtype = {'fullVisitorId': 'str', 'visitId': 'str'})
        df_list.append(dfname)
        
    df = pd.concat(df_list, ignore_index = True)
    df.to_csv(outname, index = False)
    return df

def bug_fix(df):
    drop_list = df[df['date'] == "No"].index.tolist()
    df = df.drop(drop_list)
    return df

def load_dataFrame(filename):
    print("load file " + filename + '...')
    df = pd.read_csv(filename)
    return df


def main():
    #exjson('train_split', 101)
    #concat_df('exjson_train_split', 101, 'train_df.csv')
    df = load_dataFrame('exjson_train_split/train_df.csv')
    print(len(df.index))
    print(df.columns)
    print(df['totals.transactionRevenue'].count())

#main()
