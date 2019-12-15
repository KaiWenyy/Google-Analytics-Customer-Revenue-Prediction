import csv
import os
import sys

csv.field_size_limit(sys.maxsize)
path = '/Users/Ivy/Downloads/train_v2.csv'

with open(path, 'r', newline='') as file:
    csvreader = csv.reader(file)
    a = next(csvreader)
    #print(a)
    i = j = 1
    for row in csvreader:
        print(i)
        if i <= 5000:
            #j += 1
            csv_path = os.path.join('/'.join(path.split('/')[:-1]), 'train_v2_split' + '.csv')
            # print('/'.join(path.split('/')[:-1]))
            #print(csv_path)
            # 不存在此文件的時候，就創建
            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='') as file:
                    csvwriter = csv.writer(file)
                    #csvwriter.writerow(['image_url'])
                    csvwriter.writerow(row)
                i += 1
            # 存在的時候就往裏面添加
            else:
                with open(csv_path, 'a', newline='') as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(row)
                i += 1
        else:
            break
