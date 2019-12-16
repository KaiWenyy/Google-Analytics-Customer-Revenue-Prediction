import csv
import os
import sys

csv.field_size_limit(sys.maxsize)
path = '/Users/Ivy/Desktop/datascience/final/train_v2.csv'
j = 51
loadStart = 250000
totalFiles = 100
with open(path, 'r', newline='') as file:
    csvreader = csv.reader(file)
    a = next(csvreader)
    #print(a)
    i = 1
    for row in csvreader:
        if i <= loadStart:
            i += 1
        elif i <= 5000*j:
            print(i)
            csv_path = os.path.join('/'.join(path.split('/')[:-1]), 'train_split/train_split' + str(j) + '.csv')
            #print(csv_path)
            # 不存在此文件的時候，就創建
            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='') as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(['channelGrouping','customDimensions','date','device','fullVisitorId','geoNetwork','hits','socialEngagementType','totals','trafficSource','visitId','visitNumber','visitStartTime'])
                    csvwriter.writerow(row)
                i += 1
            # 存在的時候就往裏面添加
            else:
                with open(csv_path, 'a', newline='') as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(row)
                i += 1
        if i == 5000*j:
            j += 1
        elif j >= totalFiles:
            break