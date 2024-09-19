import os,sys
import csv


def save_csv(data, path, fieldnames=['id', 'RowSpacinge', 'WordSpacinge', 'FontSize','FontShape']):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))

if __name__ == '__main__':
    label_path = r'G:\DataSet\HWDB\2.0\HWDB2.0page-label'
    label_list = os.listdir(label_path)
    all_data = []
    for label in label_list:
        RowSpacinge = label.split('-')[0]
        WordSpacinge = label.split('-')[1]
        FontSize = label.split('-')[2]

        if len(label.split('-')[3])==6:
            FontShape = label.split('-')[3][:3]
            id = label.split('-')[3][3:]
        else:
            FontShape = label.split('-')[3][:1]
            id = label.split('-')[3][1:]

        image_id_path = rf'G:\DataSet\HWDB\2.0\HWDB2.0page3\{id}'
        image_id_list = os.listdir(image_id_path)
        for image_id in image_id_list:
            all_data.append([image_id,RowSpacinge,WordSpacinge,FontSize,FontShape])
    save_csv(all_data,r'G:\DataSet\HWDB\2.0\HWDB2.0page2\styles_new.csv')