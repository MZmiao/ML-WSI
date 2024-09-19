import os,sys
import csv#导入模块
import shutil
import pandas as pd

def save_csv(data, path, fieldnames=['imagename', 'RowSpacinge', 'WordSpacinge', 'FontSize','FontShape','FontInclination','ImagePath']):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))

if __name__ == '__main__':
    # label_path = r'G:\DataSet\HWDB\2.0\HWDB2.0page-label'
    # label_list = os.listdir(label_path)
    # dic = {}
    # for i in label_list:
    #     writer_id = i.split('-')[-1]
    #     row = i.split('-')[0]
    #     world = i.split('-')[1]
    #     size = i.split('-')[2]
    #     shape = i.split('-')[3]
    #     inclination = i.split('-')[4]
    #     dic[f'{writer_id}'] =row,world,size,shape,inclination
    #
    #
    #
    #
    # img_path = r'G:\DataSet\HWDB\2.0\HWDB2.0page2\pad'
    # img_list = os.listdir(img_path)
    # all_data = []
    # for b in img_list:
    #     id = b.split('-')[0]
    #     path = img_path+'/'+b
    #     row = dic[id][0]
    #     world = dic[id][1]
    #     size = dic[id][2]
    #     shape = dic[id][3]
    #     inclination = dic[id][4]
    #     all_data.append([b,row,world,size,shape,inclination,path])
    #
    #
    # # print(all_data)
    # save_csv(all_data, r'G:\DataSet\HWDB\2.0\HWDB2.0page2\styles_new.csv')


    # id_path = r'F:\04DataSet\WritterIdentify\pad0\pad'
    # i = ['id']
    # for id in os.listdir(id_path):
    #     name = id.split('-')[0]
    #     i.append(name)
    # with open("./styles_new.csv") as csvFile:
    #     a = 0
    #     print(len(i))
    #     rows = csv.reader(csvFile)
    #     with open(("./styles_new2.csv"), 'w') as f:
    #         writer = csv.writer(f)
    #         for row in rows:
    #             row.append(i[a])
    #             writer.writerow(row)
    #             a+=1

    # import pandas as pd
    # # 读入ant-nnop.csv文件
    # df = pd.read_csv('train_new1.csv', header=None)  # 无表头
    #
    # # drop（[0]）表删除0列
    # d = df.drop([6], axis=1)
    #
    # # d为删除后得到数据，写入1.csv中
    # d.to_csv('train_new2.csv', header=False, index=False)

    with open("./styles_new.csv") as csvFile:

        with open('styles_new.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            column1 = [row[6] for row in reader]
            print(column1)
            name_list = ['id']
        for path in column1:
            if path != 'ImagePath':
                # print(path)
                name = path.split('d')[-1].split('/')[-1].split('-')[0]
                name_list.append(name)


        a = 0
        rows = csv.reader(csvFile)
        with open(("./styles_new1.csv"), 'w',newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                row.append(name_list[a])
                writer.writerow(row)
                a+=1

