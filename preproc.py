import csv
import os
import sys
import pandas as pd 
import json
import numpy as np
import pickle 

def read_and_save_data(csv_paths, df_pickle_name):

    all_df = pd.DataFrame()
    allVisitorId = {} # (key: id, value: login times)
    for path in csv_paths:
        df = pd.read_csv(path) 
        #print(df.head())  
        simple_col = ['channelGrouping', 'date', 'fullVisitorId', 'socialEngagementType',
        'visitId', 'visitNumber', 'visitStartTime']
        full_col = ['customDimensions', 'device', 'geoNetwork', 'hits', 'trafficSource', 'totals'] + simple_col
        index_range = df.index

        for col in full_col:
            if col in simple_col:
               #print('simple: ',col,  df[col].dtype)
                if col == 'fullVisitorId':
                    for index in df.index:
                        if df.at[index, col] in allVisitorId.keys():
                            allVisitorId[ df.at[index, col] ] += 1
                        else:
                            allVisitorId[ df.at[index, col] ] = 1
            else: 
                #print('notsimple: ',col, df[col].dtype)
                for index in index_range:
                    
                    if col == 'customDimensions' or col == 'hits':
                        df.at[index, col] = df.at[index, col].replace('"', "'").replace(" '", ' "').replace("['", '["').replace("{'", '{"').replace("',", '",').replace("':",'":').replace("'}",'"}').replace("']",'"]').replace("True", 'true')
                    
                    print("process", path, index, col)
                    val_dict = json.loads(df.at[index, col])
                    df.at[index, col] = val_dict

                    #drop the subject without label
                    if col == 'totals':
                        if 'transactionRevenue' not in df.at[index, col]:
                            df.drop([index], inplace = True)
                df.reset_index()

            if col == full_col[-1]:
                all_df = pd.concat([all_df,df],axis=0, ignore_index=True)

    all_df.to_pickle(df_pickle_name)
    print("save DataFrame to", df_pickle_name)
    allVisitorId.to_pickle('allVisitorId.pkl')
    print("save allVisitorId to allVisitorId.pkl")
    print("number of visitor ID:", len(allVisitorId.keys()))

def load_data(pickle_path):
    print('loading', pickle_path + ' ...') 
    df = pd.read_pickle(pickle_path)
    return df

def get_one_feature(df, column="totals", subcolumn=None, to_float=False):
    if subcolumn != None:
        data = df[column]
        container = []
        for d in data:
            container.append(d[subcolumn])
    else:
        container = df[column]
    if to_float:
        container = [float(n) for n in container]
    container = np.array(container)
    print(len(container), container[:3])
    return container


def get_hits_feature(df, subcolumn, to_float=False):
    """Features in "hits":
        'hitNumber', 'time', 'hour', 'minute', 'isInteraction', 'isEntrance', 'page', 'transaction', 
        'item', 'appInfo', 'exceptionInfo', 'product', 'promotion', 'eCommerceAction', 'experiment', 
        'customVariables', 'customDimensions', 'customMetrics', 'type', 'social', 'contentGroup', 
        'dataSource', 'publisher_infos'
    """
    data = df["hits"]  
    container = []
    if to_float:
        for d in data:  # "d" is a list containing several dicts without fixed length.
            container.append([float(e[subcolumn]) for e in d])
    else:
        for d in data:
            container.append([d[h][subcolumn] for h in len(d)])
    print(container[:3])
    container = pad_sequences(container, padding="pre", value=0.0, maxlen=30)  # TODO: improve.
    print(subcolumn, container.shape, container[:3])
    return container


def get_target(df, allVisitorId):
    target = {}
    for visitor in allVisitorId:
        target[visitor] = 0
    for index in df.index:
        visitor = df.at[index, "fullVisitorId"]
        target[visitor] += float(df.at[index, "totals"]["transactionRevenue"])
    for i, visitor in enumerate(allVisitorId):
        target[visitor] = np.log(target[visitor] + 1)
        if i < 3:
            print("VisitorID = %d, target = %.2f" % (visitor, target[visitor]))
    return target


def main():
    #csv_paths = ['train_split/train_split' + str(j+1) + '.csv' for j in range(101)]
    #read_and_save_data(csv_paths, 'df.pkl')
    df = load_data('df.pkl')
    allVisitorId = load_data('allVisitorId.pkl')
    print(len(df.index))
    print(len(allVisitorId.keys()))

main()
