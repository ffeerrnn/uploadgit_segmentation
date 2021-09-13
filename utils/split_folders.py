import cv2
import numpy as np
from skimage import io
# img = cv2.imread("/home/liufang/project/DataSets/cz2m/054528_022009.png")    #cz_202104_2m.tif
import os
import shutil

def split_folders(path, new_path, if_copy):

    file_name = os.listdir(path)
    print(len(file_name))

    parts = 2
    size = int(len(file_name) / 5)
    print(parts, size)
    for i in range(parts):
        file_new = file_name[i * size:(i + 1) * size]
        save_path = new_path + "_{}/".format(i)
        print(i, i * size, "->", (i + 1) * size, new_path)
        if not os.path.exists(save_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(save_path)
        #
        for id, f in enumerate(file_new):
            print("\t{}: {}/{}".format(i, id, len(file_new)), new_path, save_path)
            if if_copy:
                shutil.copy(os.path.join(path, f), os.path.join(save_path, f))
            else:
                shutil.move(os.path.join(path, f), os.path.join(save_path, f))


if __name__ == '__main__':
    # path = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/R-CN-JSCZS-202106-tiles"
    # new_path = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/R-CN-JSCZS-202106-tiles"
    # split_folders(path, new_path, if_copy=False)

    # path = "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles/"
    # new_path = "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles"    # NO "/"
    # split_folders(path, new_path, if_copy=False)

    path = "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_0/"
    save_path = "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_1/"
    file_name1 = os.listdir(path)
    file_name2 = os.listdir(save_path)
    print(len(file_name1), len(file_name2))
    # inter = list(set(file_name1).intersection(set(file_name2)))
    # for i, f in enumerate(inter):
    #     rf = "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_1/"+f
    #     print(i, rf)
        # os.remove(rf)
    # for i, f in enumerate(file_name):
    #     print(i, f)
    #     shutil.copy(os.path.join(path, f), os.path.join(save_path, f))



