import os
import shutil
import random

def copy_file(path1, path2, path3):
    "path1 part  to path2"

    if not os.path.exists(path2):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path2)
    if not os.path.exists(path3):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path3)

    names1 = os.listdir(path1)
    print(len(names1))
    names = random.choices(names1, k=10000)
    print(names)
    names_train = names[:7000]
    names_val = names[7000:]

    names2 = []
    # for name in names1:
    #     f = open(os.path.join(path1, name), "rb")
    #     size = len(f.read())
    #     print(file_path, size)
        # if size > 562:
        #     names2.append(name)
    # print(len(names2))
    # if len(names2) == 0:
    #     names2 = names1
    #
    # # names = random.choices(names2, k=1000)
    for index, name in enumerate(names_train):
        print(index, name)
        shutil.copy(os.path.join(path1, name), os.path.join(path2, name))
    for index, name in enumerate(names_val):
        print(index, name)
        shutil.copy(os.path.join(path1, name), os.path.join(path3, name))

    # names2 = os.listdir(path2)
    # print(len(names1), len(names2))


if __name__ == '__main__':
    path1 = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/R-CN-JSCZS-202106-tiles/"
    path2 = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/train/"
    path3 = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/val/"

    copy_file(path1, path2, path3)