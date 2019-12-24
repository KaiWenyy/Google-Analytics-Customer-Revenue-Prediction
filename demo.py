from newproc import load_other_df, load_hits_df, intersection#, bug_hit_fix, bug_other_fix


'''
OTHERS COLUMNS
	['channelGrouping', 'customDimensions.index', 'customDimensions.value',
       'date', 'device.browser', 'device.browserSize', 'device.browserVersion',
       'device.deviceCategory', 'device.flashVersion', 'device.isMobile',
       'device.language', 'device.mobileDeviceBranding',
       'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName',
       'device.mobileDeviceModel', 'device.mobileInputSelector',
       'device.operatingSystem', 'device.operatingSystemVersion',
       'device.screenColors', 'device.screenResolution', 'fullVisitorId',
       'geoNetwork.city', 'geoNetwork.cityId', 'geoNetwork.continent',
       'geoNetwork.country', 'geoNetwork.latitude', 'geoNetwork.longitude',
       'geoNetwork.metro', 'geoNetwork.networkDomain',
       'geoNetwork.networkLocation', 'geoNetwork.region',
       'geoNetwork.subContinent', 'socialEngagementType', 'totals.bounces',
       'totals.hits', 'totals.newVisits', 'totals.pageviews',
       'totals.sessionQualityDim', 'totals.timeOnSite',
       'totals.totalTransactionRevenue', 'totals.transactionRevenue',
       'totals.transactions', 'totals.visits', 'trafficSource.adContent',
       'trafficSource.adwordsClickInfo.adNetworkType',
       'trafficSource.adwordsClickInfo.criteriaParameters',
       'trafficSource.adwordsClickInfo.gclId',
       'trafficSource.adwordsClickInfo.isVideoAd',
       'trafficSource.adwordsClickInfo.page',
       'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',
       'trafficSource.campaignCode', 'trafficSource.isTrueDirect',
       'trafficSource.keyword', 'trafficSource.medium',
       'trafficSource.referralPath', 'trafficSource.source', 'visitId',
       'visitNumber', 'visitStartTime','ind']
HITS COLUMNS
       ['hits.appInfo.exitScreenName', 'hits.appInfo.landingScreenName',
       'hits.appInfo.screenName', 'hits.contentGroup.contentGroup1',
       'hits.contentGroup.contentGroup2', 'hits.contentGroup.contentGroup3',
       'hits.contentGroup.contentGroup4',
       'hits.contentGroup.contentGroupUniqueViews1',
       'hits.contentGroup.contentGroupUniqueViews2',
       'hits.contentGroup.contentGroupUniqueViews3',
       'hits.contentGroup.previousContentGroup1',
       'hits.contentGroup.previousContentGroup2',
       'hits.contentGroup.previousContentGroup3',
       'hits.contentGroup.previousContentGroup4',
       'hits.contentGroup.previousContentGroup5', 'hits.dataSource',
       'hits.eCommerceAction.action_type', 'hits.eCommerceAction.option',
       'hits.eCommerceAction.step', 'hits.eventInfo.eventAction',
       'hits.eventInfo.eventCategory', 'hits.eventInfo.eventLabel',
       'hits.exceptionInfo.isFatal', 'hits.hitNumber', 'hits.hour',
       'hits.isEntrance', 'hits.isExit', 'hits.isInteraction',
       'hits.item.currencyCode', 'hits.item.transactionId', 'hits.minute',
       'hits.page.hostname', 'hits.page.pagePath', 'hits.page.pagePathLevel1',
       'hits.page.pagePathLevel2', 'hits.page.pagePathLevel3',
       'hits.page.pagePathLevel4', 'hits.page.pageTitle', 'hits.promoId',
       'hits.promoName', 'hits.promotionActionInfo.promoIsClick',
       'hits.promotionActionInfo.promoIsView', 'hits.referer',
       'hits.social.hasSocialSourceReferral', 'hits.social.socialNetwork',
       'hits.time', 'hits.transaction.affiliation',
       'hits.transaction.currencyCode',
       'hits.transaction.localTransactionRevenue',
       'hits.transaction.localTransactionShipping',
       'hits.transaction.localTransactionTax',
       'hits.transaction.transactionCoupon', 'hits.transaction.transactionId',
       'hits.transaction.transactionRevenue',
       'hits.transaction.transactionShipping',
       'hits.transaction.transactionTax', 'hits.type',
       'hits.v2ProductCategory', 'hits.v2ProductName', 'ind']
'''
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import time
import datetime
import pandas as pd
raw_df_train = load_other_df('test_train/others/train_df2.csv')
raw_df_test = load_other_df('test_train/others/test_df2.csv')
test_df = raw_df_test
train_df = raw_df_train
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.impute import SimpleImputer
#np.set_printoptions(threshold=np.inf)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#pd.set_option('max_colwidth',100)
#lb = LabelBinarizer()
#lb.fit(train_df['channelGrouping'])
#transformed = lb.transform(train_df['channelGrouping'])
#ohe_df = pd.DataFrame(transformed)
#train_df = pd.concat([train_df, ohe_df], axis=1).drop(['channelGrouping'], axis=1)
#for key in train_df:
#    print(key)
print(train_df.isna().sum())
train_df = train_df.replace('not available in demo dataset', np.nan)
test_df = test_df.replace('not available in demo dataset', np.nan)

train_df['customDimensions.index'] = train_df['customDimensions.index'].replace(4, 1)
train_df['customDimensions.index'] = train_df['customDimensions.index'].replace(np.nan, 0)
test_df['customDimensions.index'] = test_df['customDimensions.index'].replace(4, 1)
test_df['customDimensions.index'] = test_df['customDimensions.index'].replace(np.nan, 0)


train_df['device.isMobile'] = train_df['device.isMobile'].replace(True, 1)
train_df['device.isMobile'] = train_df['device.isMobile'].replace(False, 0)
test_df['device.isMobile'] = test_df['device.isMobile'].replace(True, 1)
test_df['device.isMobile'] = test_df['device.isMobile'].replace(False, 0)

train_df['totals.bounces'] = train_df['totals.bounces'].replace(np.nan, 0)
test_df['totals.bounces'] = test_df['totals.bounces'].replace(np.nan, 0)

train_df['totals.newVisits'] = train_df['totals.newVisits'].replace(np.nan, 0)
test_df['totals.newVisits'] = test_df['totals.newVisits'].replace(np.nan, 0)

train_df['trafficSource.adwordsClickInfo.isVideoAd'] = train_df['trafficSource.adwordsClickInfo.isVideoAd'].replace(False, 0)
train_df['trafficSource.adwordsClickInfo.isVideoAd'] = train_df['trafficSource.adwordsClickInfo.isVideoAd'].replace(np.nan, 1)
test_df['trafficSource.adwordsClickInfo.isVideoAd'] = test_df['trafficSource.adwordsClickInfo.isVideoAd'].replace(False, 0)
test_df['trafficSource.adwordsClickInfo.isVideoAd'] = test_df['trafficSource.adwordsClickInfo.isVideoAd'].replace(np.nan, 1)


train_df['trafficSource.isTrueDirect'] = train_df['trafficSource.isTrueDirect'].replace(True, 1)
train_df['trafficSource.isTrueDirect'] = train_df['trafficSource.isTrueDirect'].replace(np.nan, 0)
test_df['trafficSource.isTrueDirect'] = test_df['trafficSource.isTrueDirect'].replace(True, 1)
test_df['trafficSource.isTrueDirect'] = test_df['trafficSource.isTrueDirect'].replace(np.nan, 0)

#total rows_num = 502870 (drop all unvailable)
drop = []
for key in train_df:
    if(train_df[key].isna().sum() == 502870):
        drop.append(key)
train_df = train_df.drop(drop, axis=1)
test_df = test_df.drop(drop, axis=1)

#start_day = datetime.datetime.strptime('20160804','%Y%m%d') #start 2016 08 04
#for key in train_df:
#    print(train_df[key].unique())
#    print(len(train_df[key].unique()))
train_df = train_df.sort_values(by=['visitStartTime'], ascending=False)
test_df = test_df.sort_values(by=['visitStartTime'], ascending=False)

train_df['new_date'] = [datetime.datetime.strptime(str(d),'%Y%m%d') for d in train_df['date']]
train_df['new_date'] = [d.weekday() for d in train_df['new_date']]
test_df['new_date'] = [datetime.datetime.strptime(str(d),'%Y%m%d') for d in test_df['date']]
test_df['new_date'] = [d.weekday() for d in test_df['new_date']]
#train_df['date - start'] = train_df['new_date'] - start_day
#print(train_df['device.browser'].value_counts())
values = train_df['device.browser'].value_counts().keys().tolist()
counts = train_df['device.browser'].value_counts().tolist()
#print(values)
#print(counts)
maxkv = values[0:10]
#print(train_df['new_date'].value_counts())
#print(train_df['date - start'].value_counts())
#train_df = train_df.drop(['date'], axis = 1)
#test_df = test_df.drop(['date'], axis = 1)

for i in maxkv:
    values.remove(i)
for i in values:
    train_df['device.browser'] = train_df['device.browser'].replace(i, 'others')
    test_df['device.browser'] = test_df['device.browser'].replace(i, 'others')
#print(train_df['device.operatingSystem'].value_counts())
values_do = train_df['device.operatingSystem'].value_counts().keys().tolist()
counts_do = train_df['device.operatingSystem'].value_counts().tolist()
maxkdo = values_do[0:7]
for i in maxkdo:
    values_do.remove(i)
for i in values_do:
    train_df['device.operatingSystem'] = train_df['device.operatingSystem'].replace(i, 'others')
    test_df['device.operatingSystem'] = test_df['device.operatingSystem'].replace(i, 'others')
dic = {}
dic_test = {}

values_fvid = train_df['fullVisitorId'].value_counts().keys().tolist()
count_fvid = train_df['fullVisitorId'].value_counts().tolist()
values_fvid_test = test_df['fullVisitorId'].value_counts().keys().tolist()
count_fvid_test = test_df['fullVisitorId'].value_counts().tolist()

for i in range(len(values_fvid)):
    dic[values_fvid[i]] = count_fvid[i]
for i in range(len(values_fvid_test)):
    dic_test[values_fvid_test[i]] = count_fvid_test[i]
    
section_num = []
for fvid in train_df['fullVisitorId']:
    section_num.append(dic[fvid])
    dic[fvid] = dic[fvid] - 1
train_df['section_num'] = section_num
section_num_test = []
for fvid in test_df['fullVisitorId']:
    section_num_test.append(dic_test[fvid])
    dic_test[fvid] = dic_test[fvid] - 1
test_df['section_num'] = section_num_test
#for key in train_df:
#    print(key)
#    print(train_df[key].unique())
#    print(len(train_df[key].unique()))
values_sc = train_df['geoNetwork.subContinent'].value_counts().keys().tolist()
counts_sc = train_df['geoNetwork.subContinent'].value_counts().tolist()
maxksc = values_sc[0:13]
for i in maxksc:
    values_sc.remove(i)
for i in values_sc:
    train_df['geoNetwork.subContinent'] = train_df['geoNetwork.subContinent'].replace(i, 'others')
    test_df['geoNetwork.subContinent'] = test_df['geoNetwork.subContinent'].replace(i, 'others')

train_df = train_df.drop(['socialEngagementType'], axis=1)
train_df = train_df.drop(['totals.visits'], axis=1)
train_df = train_df.drop(['visitId'], axis=1)
test_df = test_df.drop(['socialEngagementType'], axis=1)
test_df = test_df.drop(['totals.visits'], axis=1)
test_df = test_df.drop(['visitId'], axis=1)

train_df = train_df.drop(['geoNetwork.networkDomain'], axis=1)
train_df = train_df.drop(['trafficSource.adwordsClickInfo.gclId'], axis=1)
train_df = train_df.drop(['trafficSource.keyword'], axis=1)
train_df = train_df.drop(['trafficSource.referralPath'], axis=1)
test_df = test_df.drop(['geoNetwork.networkDomain'], axis=1)
test_df = test_df.drop(['trafficSource.adwordsClickInfo.gclId'], axis=1)
test_df = test_df.drop(['trafficSource.keyword'], axis=1)
test_df = test_df.drop(['trafficSource.referralPath'], axis=1)

#for key in train_df:
#    print(key)
#    print(train_df[key].unique())
#    print(len(train_df[key].unique()))
train_df = train_df.drop(['totals.sessionQualityDim'], axis=1)
test_df = test_df.drop(['totals.sessionQualityDim'], axis=1)

x = train_df["totals.pageviews"]
x = x.to_numpy()
x = x.reshape(-1, 1)
x1 = test_df["totals.pageviews"]
x1 = x1.to_numpy()
x1 = x1.reshape(-1, 1)

train_df["totals.pageviews"] = SimpleImputer(missing_values=np.nan, strategy="mean").fit(x).transform(x)
test_df["totals.pageviews"] = SimpleImputer(missing_values=np.nan, strategy="mean").fit(x).transform(x1)   ##

train_df["totals.timeOnSite"].fillna(0, inplace=True)
train_df["totals.transactionRevenue"].fillna(0, inplace=True)
train_df["totals.totalTransactionRevenue"].fillna(0, inplace=True)
train_df["totals.transactions"].fillna(0, inplace=True)
train_df["trafficSource.adwordsClickInfo.page"].fillna(0, inplace=True)
test_df["totals.timeOnSite"].fillna(0, inplace=True)
test_df["totals.transactions"].fillna(0, inplace=True)
test_df["trafficSource.adwordsClickInfo.page"].fillna(0, inplace=True)




#print(train_df.isna().sum())
for key in train_df:
    print(key)
    print(train_df[key].unique())
    print(len(train_df[key].unique()))
print(train_df.isna().sum())

train_df = train_df.sort_values(by=['ind'])
test_df = test_df.sort_values(by=['ind'])

train_y = train_df["totals.transactionRevenue"].to_numpy()
np.save("train_y", train_y)

cat_cols_need_emb = ['channelGrouping', 'customDimensions.value',
                     'device.browser', 'device.deviceCategory',
                     'device.operatingSystem', 'geoNetwork.continent',
                     'geoNetwork.subContinent', 'trafficSource.adwordsClickInfo.adNetworkType',
                     'trafficSource.adwordsClickInfo.slot',
                     'trafficSource.campaign', 'trafficSource.medium',
                     'new_date']

cat_cols_noneed_emb = ['customDimensions.index', 'device.isMobile',
                       'totals.bounces', 'totals.newVisits',
                       'trafficSource.adwordsClickInfo.isVideoAd',
                       'trafficSource.isTrueDirect']

cat_cols_need_outeremb = ['geoNetwork.city', 'geoNetwork.country',
                          'geoNetwork.metro', 'geoNetwork.region',
                          'trafficSource.adContent', 'trafficSource.source']

num_col = ['totals.hits', 'totals.pageviews', 'totals.timeOnSite',
           'totals.transactions', 'trafficSource.adwordsClickInfo.page',
           'visitNumber', 'visitStartTime', 'section_num']

all_select = ['channelGrouping', 'customDimensions.index',
              'customDimensions.value', 'device.browser',
              'device.deviceCategory', 'device.isMobile',
              'device.operatingSystem', 'geoNetwork.city',
              'geoNetwork.continent', 'geoNetwork.country',
              'geoNetwork.metro', 'geoNetwork.region',
              'geoNetwork.subContinent', 'totals.bounces',
              'totals.hits', 'totals.newVisits', 'totals.pageviews',
              'totals.timeOnSite', 'totals.transactions',
              'trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType',
              'trafficSource.adwordsClickInfo.isVideoAd',
              'trafficSource.adwordsClickInfo.page',
              'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',
              'trafficSource.isTrueDirect', 'trafficSource.medium', 'trafficSource.source',
              'visitNumber', 'visitStartTime', 'new_date', 'section_num']
train_df = train_df[all_select]
test_df = test_df[all_select]
train_df = train_df.replace(np.nan, 'unk')
test_df = test_df.replace(np.nan, 'unk')
#export_csv = train_df.to_csv (r'C:\Users\美芬\Desktop\data_science_final\export_dataframe.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
#export_csv_test = test_df.to_csv (r'C:\Users\美芬\Desktop\data_science_final\export_dataframe_test.csv', index = None, header=True)

#lb = LabelBinarizer()
#lb.fit(train_df['customDimensions.value'])
#transformed = lb.transform(train_df['customDimensions.value'])
#ohe_df = pd.DataFrame(transformed)
#train_df = pd.concat([train_df, ohe_df], axis=1).drop(['customDimensions.value'], axis=1)
#transformed_test = lb.transform(test_df['customDimensions.value'])
#ohe_df_test = pd.DataFrame(transformed_test)
#test_df = pd.concat([test_df, ohe_df_test], axis=1).drop(['customDimensions.value'], axis=1)

#finished
for key in cat_cols_need_outeremb:
    dic = {}
    dic['unk'] = 0
    dic['unseen'] = 1
    count = 2
    l = train_df[key].value_counts().keys().tolist()
    l = [x for x in l if str(x) != 'unk']
    for item in l:
        dic[item] = count
        count += 1
    train_df[key] = [dic[x] for x in train_df[key]]
    test_df[key] = [dic[x] if x in dic else 1 for x in test_df[key]]

for key in num_col:
    enc = StandardScaler()
    x = train_df[key]
    x = x.to_numpy()
    x = x.reshape(-1, 1)
    x1 = test_df[key]
    x1 = x1.to_numpy()
    x1 = x1.reshape(-1, 1)
    enc.fit(x)
    train_df[key] = enc.transform(x)
    test_df[key] = enc.transform(x1)


for key in cat_cols_need_emb:
    enc = OneHotEncoder(handle_unknown='ignore')
    e = train_df[key].value_counts().keys().tolist()
    e = [x for x in e if str(x) != 'unk']
    e = np.array(e).reshape(-1, 1)
    enc.fit(e)
    transformed = enc.transform(train_df[key].values.reshape(-1,1)).toarray()
    ohe_df = pd.DataFrame(transformed, columns = [str(key)+str(int(i)) for i in range(transformed.shape[1])])
    train_df = pd.concat([train_df, ohe_df], axis=1).drop([key], axis=1)
    transformed_test = enc.transform(test_df[key].values.reshape(-1,1)).toarray()
    ohe_df_test = pd.DataFrame(transformed_test, columns = [str(key)+str(int(i)) for i in range(transformed_test.shape[1])])
    test_df = pd.concat([test_df, ohe_df_test], axis=1).drop([key], axis=1)

for key in test_df:
    test_df[key] = test_df[key].astype(float)

for key in train_df:
    train_df[key] = train_df[key].astype(float)

x2_train = train_df[cat_cols_need_outeremb].to_numpy() #outer embedding
x2_test = test_df[cat_cols_need_outeremb].to_numpy()
np.save("x2_train", x2_train)
np.save("x2_test", x2_test)


x1_train = train_df.drop(cat_cols_need_outeremb, axis=1).to_numpy()
x1_test = test_df.drop(cat_cols_need_outeremb, axis=1).to_numpy()
np.save("x1_train", x1_train)
np.save("x1_test", x1_test)

print("x2_train.shape()")
print(x2_train.shape)
print("x2_test.shape()")
print(x2_test.shape)
print("x1_train.shape()")
print(x1_train.shape)
print("x1_test.shape()")
print(x1_test.shape)
print("train_y.shape()")
print(train_y.shape)


#okay
#enc = OneHotEncoder(handle_unknown='ignore')
#e = train_df['customDimensions.value'].value_counts().keys().tolist()
#e = [x for x in e if str(x) != 'unk']
#e = np.array(e).reshape(-1, 1)
#enc.fit(e)
#transformed = enc.transform(train_df['customDimensions.value'].values.reshape(-1,1)).toarray()
#ohe_df = pd.DataFrame(transformed, columns = ["customDimensions.value_"+str(int(i)) for i in range(transformed.shape[1])])
#train_df = pd.concat([train_df, ohe_df], axis=1).drop(['customDimensions.value'], axis=1)
#transformed_test = enc.transform(test_df['customDimensions.value'].values.reshape(-1,1)).toarray()
#ohe_df_test = pd.DataFrame(transformed_test, columns = ["customDimensions.value_"+str(int(i)) for i in range(transformed_test.shape[1])])
#test_df = pd.concat([test_df, ohe_df_test], axis=1).drop(['customDimensions.value'], axis=1)


export_csv = train_df.to_csv (r'C:\Users\美芬\Desktop\data_science_final\export_dataframe.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
export_csv_test = test_df.to_csv (r'C:\Users\美芬\Desktop\data_science_final\export_dataframe_test.csv', index = None, header=True)










