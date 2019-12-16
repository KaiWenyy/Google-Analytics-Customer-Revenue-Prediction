import csv
import os
import sys
import pandas as pd 
import json
import numpy as np 

def read_data():
    #df = pd.read_csv('train_v2_split.csv')
    df = pd.read_csv('train_5000r.csv')  
    #print(df.head())  
    simple_col = ['channelGrouping', 'date', 'fullVisitorId', 'socialEngagementType',
    'visitId', 'visitNumber', 'visitStartTime']
    full_col = ['customDimensions', 'device', 'geoNetwork', 'hits', 'trafficSource', 'totals'] + simple_col
    allVisitorId = {} # (key: id, value: login times)
    allchannelGroup = df.channelGrouping.unique()
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
            #pass
            #print('notsimple: ',col, df[col].dtype)
            for index in index_range:
                
                if col == 'customDimensions' or col == 'hits':
                    df.at[index, col] = df.at[index, col].replace(" '", ' "').replace("['", '["').replace("{'", '{"').replace("',", '",').replace("':",'":').replace("'}",'"}').replace("']",'"]').replace("True", 'true')
                    #.replace('Men"s',"Men's").replace('Women"s',"Women's").replace('Kid"s',"Kid's").replace('Kids"',"Kids'")
                #print(index, col)
                val_dict = json.loads(df.at[index, col])
                df.at[index, col] = val_dict

                #drop the subject without label
                if col == 'totals':
                    if 'transactionRevenue' not in df.at[index, col]:
                        #pass
                        df.drop([index], inplace = True)

            df.reset_index()
    print(len(allVisitorId.keys()))
    print(allchannelGroup)
    #print(allVisitorId)
    return df, allVisitorId


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
    df, allVisitorId = read_data()






