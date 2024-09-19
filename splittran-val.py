import argparse
import csv
import numpy as np
from tqdm import tqdm


def save_csv(data, path, fieldnames=['imagename','image_path', 'RowSpacinge', 'WordSpacinge', 'FontSize','FontShape','FontInclination']):
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
    train = []
    val = []
    #打开styles.csv文件获取数据信息
    with open(annotation) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in tqdm(reader, total=reader.line_num):
            img_id = row['imagename'].split('P')[1][:2]
            # print(img_id)
            RowSpacinge = row['RowSpacinge']
            WordSpacinge = row['WordSpacinge']
            FontSize = row['FontSize']
            FontShape = row['FontShape']
            img_name = row['ImagePath']
            FontInclination = row["FontInclination"]
            # img_name = os.path.join(input_folder, 'images', str(img_id))
            # if os.path.exists(img_name):
            #     img = Image.open(img_name)
            #     if img.mode == "RGBA":
            #         all_data.append([img_name, RowSpacinge, WordSpacinge, FontSize, FontShape])

            all_data.append([img_name, RowSpacinge, WordSpacinge, FontSize, FontShape,FontInclination])
            if img_id in ['16','17','18','19']:
                train.append([img_name, RowSpacinge, WordSpacinge, FontSize, FontShape,FontInclination])
            if img_id == '20':
                val.append([img_name, RowSpacinge, WordSpacinge, FontSize, FontShape,FontInclination])

            # img_name = '\mnt'+'\HWDB2.0page2\images'+f'\{str(img_id)}' + '.png'
            # all_data.append([img_name, RowSpacinge, WordSpacinge, FontSize, FontShape])
            # print(all_data)
    np.random.seed(18)
    train  = np.asarray(train)
    val = np.asarray(val)
    inds1 = np.random.choice(1674,1674,replace=False)
    inds2 = np.random.choice(418, 418, replace=False)
    save_csv(train[inds1][:], r'G:\DataSet\HWDB\2.0\HWDB2.0page2\train_new.csv')
    save_csv(val[inds2][:], r'G:\DataSet\HWDB\2.0\HWDB2.0page2\val_new.csv')

    # # 设置随机数生成器的种子，以便我们稍后可以重现结果
    # np.random.seed(42)
    # # 从列表中构造Numpy数组
    # all_data = np.asarray(all_data)
    # # 随机抽取40000个样本
    # inds = np.random.choice(8368, 8368, replace=False)
    # 将数据拆分为train/val，并将其保存为csv文件
    # save_csv(all_data[inds][:6694],  r'G:\DataSet\HWDB\2.0\HWDB2.0page2\train_new.csv')
    # save_csv(all_data[inds][6694:], r'G:\DataSet\HWDB\2.0\HWDB2.0page2\val_new.csv')
    # # # save_csv(all_data[inds][:1477],  r'.\train_new.csv')
    # # save_csv(all_data[inds][1477:2092], r'.\val_new.csv')
