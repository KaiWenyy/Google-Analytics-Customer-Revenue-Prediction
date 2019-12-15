import csv
import os
import sys
import pandas as pd 
import json

df = pd.read_csv('train_v2_split.csv')  
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
            print(index, col)
            val_dict = json.loads(df.at[index, col])
            df.at[index, col] = val_dict

            #drop the subject without label
            if col == 'totals':
                if 'transactionRevenue' in df.at[index, col]:
                    #pass
                    df.drop([index],inplace = True)

        df.reset_index()



print(len(allVisitorId.keys()))
print(allchannelGroup)
#print(allVisitorId)