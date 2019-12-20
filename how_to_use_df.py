from newproc import load_other_df, load_hits_df, bug_hit_fix

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
       'visitNumber', 'visitStartTime']

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

#train
train_df = load_other_df('test_train/others/train_df.csv')
nrows = len(train_df.index)
print(nrows)
print(train_df['totals.transactionRevenue'].count())

train_hit_df = load_hits_df('test_train/hits/train_df.csv')
train_hit_df = bug_hit_fix(train_hit_df)  # drop the row which colun 'ind' = nan
print(train_hit_df.columns)
print(len(train_hit_df.index))
hit_nrows = len(train_hit_df['ind'].unique())
print( hit_nrows )
#51579 1 5
assert hit_nrows == nrows


#test
test_df = load_other_df('test_train/others/test_select_df.csv')
nrows = len(test_df.index)
print(nrows)
print(test_df['totals.transactionRevenue'].count())

test_hit_df = load_hits_df('test_train/hits/test_select_df.csv')
print(len(test_hit_df.index))
hit_nrows = len(test_hit_df['ind'].unique())
print( hit_nrows )

assert hit_nrows == nrows






