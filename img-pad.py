import os
from PIL import Image
import cv2

org_img_path = r'G:\DataSet\HWDB\2.0\HWDB2.0page2\images'
shapelist = []
list = os.listdir(org_img_path)
for i in list:
    org_iamge = cv2.imread(org_img_path + '/' + i)

    # x = org_iamge.shape[0]
    # pading_size = int((2483-x)/2)
    # new_iamge = cv2.copyMakeBorder(org_iamge,pading_size,pading_size,pading_size,pading_size,borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
    # print(i, pading_size, new_iamge.shape)
    # new_iamge = cv2.resize(new_iamge,(2480,2480))
    # cv2.imwrite(rf'G:\DataSet\HWDB\2.0\HWDB2.0page2\111\{i}',new_iamge)

    x1 = cv2.resize(org_iamge[:1240,:1240],(448,448))
    cv2.imwrite(rf'G:\DataSet\HWDB\2.0\HWDB2.0page2\000\{i.split(".")[0]}-1.png', x1)
    x2 = cv2.resize(org_iamge[:1240,1040:2280],(448,448))
    cv2.imwrite(rf'G:\DataSet\HWDB\2.0\HWDB2.0page2\000\{i.split(".")[0]}-2.png', x2)
    x3 = cv2.resize(org_iamge[1040:2280, :1240], (448,448))
    cv2.imwrite(rf'G:\DataSet\HWDB\2.0\HWDB2.0page2\000\{i.split(".")[0]}-3.png', x3)
    x4 = cv2.resize(org_iamge[1040:2280,1040:2280],(448,448))
    cv2.imwrite(rf'G:\DataSet\HWDB\2.0\HWDB2.0page2\000\{i.split(".")[0]}-4.png', x4)
    print(i)
