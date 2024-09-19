import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class AttributesDataset():
    def __init__(self, annotation_path):
        RowSpacing_labels = []
        WordSpacing_labels = []
        FontSize_labels = []
        FontShape_labels = []
        FontInclination_labels = []
        id_labels = []


        #打开csv文件读取标签
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                RowSpacing_labels.append(row['RowSpacinge'])
                WordSpacing_labels.append(row['WordSpacinge'])
                FontSize_labels.append(row['FontSize'])
                FontShape_labels.append(row['FontShape'])
                FontInclination_labels.append((row['FontInclination']))
                id_labels.append((row['FontInclination']))

        #读取不重复的标签
        self.RowSpacing_labels = np.unique(RowSpacing_labels)
        self.WordSpacing_labels = np.unique(WordSpacing_labels)
        self.FontSize_labels = np.unique(FontSize_labels)
        self.FontShape_labels = np.unique(FontShape_labels)
        self.FontInclination_labels = np.unique(FontInclination_labels)
        self.FontInclination_labels = np.unique(id_labels)


        #读取标签的数量
        self.num_RowSpacing = len(self.RowSpacing_labels)
        self.num_WordSpacing = len(self.WordSpacing_labels)
        self.num_FontSize = len(self.FontSize_labels)
        self.num_FontShape = len(self.FontShape_labels)
        self.num_FontInclination = len(self.FontInclination_labels)
        self.num_FontInclination = len(self.id_labels)


        #标签id和名称之间的互相转化
        self.RowSpacing_id_to_name = dict(zip(range(len(self.RowSpacing_labels)), self.RowSpacing_labels))
        self.RowSpacing_name_to_id = dict(zip(self.RowSpacing_labels, range(len(self.RowSpacing_labels))))

        self.WordSpacing_id_to_name = dict(zip(range(len(self.WordSpacing_labels)), self.WordSpacing_labels))
        self.WordSpacing_name_to_id = dict(zip(self.WordSpacing_labels, range(len(self.WordSpacing_labels))))

        self.FontSize_id_to_name = dict(zip(range(len(self.FontSize_labels)), self.FontSize_labels))
        self.FontSize_name_to_id = dict(zip(self.FontSize_labels, range(len(self.FontSize_labels))))

        self.FontShape_id_to_name = dict(zip(range(len(self.FontShape_labels)), self.FontShape_labels))
        self.FontShape_name_to_id = dict(zip(self.FontShape_labels, range(len(self.FontShape_labels))))

        self.FontInclination_id_to_name = dict(zip(range(len(self.FontInclination_labels)),self.FontInclination_labels))
        self.FontInclination_name_to_id = dict(zip(self.FontInclination_labels, range(len(self.FontInclination_labels))))

        self.id_id_to_name = dict(zip(range(len(self.id_labels)), self.id_labels))
        self.id_name_to_id = dict(zip(self.id_labels, range(len(self.id_labels))))

class FashionDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # 初始化阵列以存储地面实况标签和图像路径
        self.data = []
        self.RowSpacing_labels = []
        self.WordSpacing_labels = []
        self.FontSize_labels = []
        self.FontShape_labels = []
        self.FontInclination_labels = []
        self.id_labels = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.RowSpacing_labels .append(self.attr.RowSpacing_name_to_id[row['RowSpacinge']])
                self.WordSpacing_labels.append(self.attr.WordSpacing_name_to_id[row['WordSpacinge']])
                self.FontSize_labels.append(self.attr.FontSize_name_to_id[row['FontSize']])
                self.FontShape_labels.append(self.attr.FontShape_name_to_id[row['FontShape']])
                self.FontInclination_labels.append(self.attr.FontInclination_name_to_id[row['FontInclination']])
                self.id_labels.append(self.attr.id_name_to_id[row['id']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 按索引获取数据样本
        img_path = self.data[idx]

        # 读取图像、将图像从RGBA转换为RGB，改变图像的大小
        img = Image.open(img_path)
        # img = img.convert('RGB')
        img = img.resize((1344,1344))

        # 应用图像增强
        if self.transform:
            img = self.transform(img)

        # 返回图像和所有关联的标签
        dict_data = {
            'img': img,
            'labels': {
                'RowSpacing_labels': self.RowSpacing_labels[idx],
                'WordSpacing_labels': self.WordSpacing_labels[idx],
                'FontSize_labels': self.FontSize_labels[idx],
                'FontShape_labels':self.FontShape_labels[idx],
                'FontInclination_labels':self.FontInclination_labels[idx],
                'id_labels': self.id_labels[idx]
            }
        }
        return dict_data
