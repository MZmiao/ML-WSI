import argparse
import csv
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def save_csv(data, path, fieldnames=['image_path', 'RowSpacinge', 'WordSpacinge', 'FontSize','FontShape']):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split data for the dataset')

    #数据集所在路径和保存标签信息路径
    parser.add_argument('--input', type=str, default=r"G:\DataSet\HWDB\2.0\HWDB2.0page2", help="Path to the dataset")
    parser.add_argument('--output', type=str, default="", help="Path to the working folder")

    args = parser.parse_args()
    input_folder = args.input
    output_folder = args.output
    #所有图像标签保存的文件名称
    annotation = r'G:\DataSet\HWDB\2.0\HWDB2.0page2\styles_new.csv'
    # annotation = r'\mnt\HWDB2.0page2\styles_new.csv'


    all_data = []
    #打开styles.csv文件获取数据信息
    with open(annotation) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in tqdm(reader, total=reader.line_num):
            # img_id = row['id']
            RowSpacinge = row['RowSpacinge']
            WordSpacinge = row['WordSpacinge']
            FontSize = row['FontSize']
            FontShape = row['FontShape']
            img_name = row['ImagePath']
            # img_name = os.path.join(input_folder, 'images', str(img_id))
            # if os.path.exists(img_name):
            #     img = Image.open(img_name)
            #     if img.mode == "RGBA":
            #         all_data.append([img_name, RowSpacinge, WordSpacinge, FontSize, FontShape])
            all_data.append([img_name, RowSpacinge, WordSpacinge, FontSize, FontShape])

            # img_name = '\mnt'+'\HWDB2.0page2\images'+f'\{str(img_id)}' + '.png'
            # all_data.append([img_name, RowSpacinge, WordSpacinge, FontSize, FontShape])
            # print(all_data)
    # 设置随机数生成器的种子，以便我们稍后可以重现结果
    np.random.seed(42)
    # 从列表中构造Numpy数组
    all_data = np.asarray(all_data)
    # 随机抽取40000个样本
    inds = np.random.choice(8368, 8368, replace=False)
    # 将数据拆分为train/val，并将其保存为csv文件
    # save_csv(all_data[inds][:6694],  r'G:\DataSet\HWDB\2.0\HWDB2.0page2\train_new.csv')
    # save_csv(all_data[inds][6694:], r'G:\DataSet\HWDB\2.0\HWDB2.0page2\val_new.csv')
    # # # save_csv(all_data[inds][:1477],  r'.\train_new.csv')
    # # save_csv(all_data[inds][1477:2092], r'.\val_new.csv')
