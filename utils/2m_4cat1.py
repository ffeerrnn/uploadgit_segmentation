import os
import cv2
import numpy as np


def _2m_4cat1(path, path1, path2):
    # from 4 pics to 1 pic
    # path = "/home/liufang/project/DataSets/cz/EPSG_4326_15/213_086_result_classes/"
    # path1 = "/home/liufang/project/DataSets/cz/EPSG_4326_15_result_result_classes/"
    # path2 = "/home/liufang/project/DataSets/cz/cat_EPSG_4326_15_classes/"
    # names = os.listdir(path)
    # names1 = os.listdir(path1)
    # print(len(names))

    # path = "/home/liufang/project/DataSets/cz2m/{}/R-CN-JSCZS-{}-2M-tiles_result_classes/".format(date, date)    # R-CN-JSCZS-{}-2M-tiles_result_classes
    # path1 = "/home/liufang/project/DataSets/cz2m/{}/R-CN-JSCZS-{}-2M-tiles_result_classes/".format(date, date)
    # path2 = "/home/liufang/project/DataSets/cz2m/{}/cat_R-CN-JSCZS-{}-2M-tiles_result_classes/".format(date, date)

    if not os.path.exists(path2):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path2)

    names = os.listdir(path)
    names1 = []    # base
    names2 = []    # d lr r
    for n in names:
        if len(n.split("_")) == 2:
            names1.append(n)
        else:
            names2.append(n)
    assert len(names1)*3 == len(names2), print(len(names1), len(names2))

    print(len(names))

    for i ,name in enumerate(names1):
        print(i,"/",len(names1))
        l = cv2.imread(os.path.join(path, name))
        ld = cv2.imread(os.path.join(path1, name.split(".")[0] + "_lr.png"))    # 右下    "_ld.png"
        r = cv2.imread(os.path.join(path1, name.split(".")[0] + "_r.png"))    # 右
        d = cv2.imread(os.path.join(path1, name.split(".")[0] + "_d.png"))    # 下
        img = np.zeros_like(l)
        img[:128, :128, :] = l[64:192, 64:192, :]
        img[128:, :128, :] = d[64:192, 64:192, :]
        img[:128, 128:, :] = r[64:192, 64:192, :]
        img[128:, 128:, :] = ld[64:192, 64:192, :]

        cv2.imwrite(os.path.join(path2, name), img)



    names = os.listdir(path)
    print(len(names))


if __name__ == '__main__':
    # date = "202101"
    # path = "/home/liufang/project/DataSets/cz2m/{}/R-CN-JSCZS-{}-2M-tiles_result_classes/".format(date, date)  # R-CN-JSCZS-{}-2M-tiles_result_classes
    # path1 = "/home/liufang/project/DataSets/cz2m/{}/R-CN-JSCZS-{}-2M-tiles_result_classes/".format(date, date)
    # path2 = "/home/liufang/project/DataSets/cz2m/{}/cat_R-CN-JSCZS-{}-2M-tiles_result_classes/".format(date, date)
    # _2m_4cat1(path, path1, path2)


    path = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/6R-CN-JSCZS-202106-tiles_result_classes/"
    path1 = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/6R-CN-JSCZS-202106-tiles_result_classes/"
    path2 = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/6cat_R-CN-JSCZS-202106-tiles_result_classes/"
    _2m_4cat1(path, path1, path2)


