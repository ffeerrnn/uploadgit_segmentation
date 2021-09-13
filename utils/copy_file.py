import os
import shutil

def copy_file(path1, path2):
    "path1 to path2"
    if isinstance(path1, list):
        names1 = []
        for p in path1:
            names1.extend(os.listdir(p))
    else:
        names1 = os.listdir(path1)

    names2 = os.listdir(path2)
    print(len(names1), len(names2))
    #
    # names = list(set(names1) - set(names2))
    # for name in names:
    #     shutil.copy(os.path.join(path1, name), os.path.join(path2, name))
    #
    # names2 = os.listdir(path2)
    # print(len(names1), len(names2))


if __name__ == '__main__':
    date = "202101"
    path1 = "/home/liufang/project/DataSets/cz2m/{}/R-CN-JSCZS-{}-2M-tiles".format(date, date)
    path2 = "/home/liufang/project/DataSets/cz2m/{}/R-CN-JSCZS-{}-2M-tiles_result_classes".format(date, date)

    path1 = ["/home/liufang/Files/常州市影像图/2021年上半年亚米图/R-CN-JSCZS-202106-tiles_0/",
             "/home/liufang/Files/常州市影像图/2021年上半年亚米图/R-CN-JSCZS-202106-tiles_1/",
             "/home/liufang/Files/常州市影像图/2021年上半年亚米图/R-CN-JSCZS-202106-tiles_2/",
             "/home/liufang/Files/常州市影像图/2021年上半年亚米图/R-CN-JSCZS-202106-tiles_3/",
             "/home/liufang/Files/常州市影像图/2021年上半年亚米图/R-CN-JSCZS-202106-tiles_4/"]
    path2 = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/6R-CN-JSCZS-202106-tiles_result_classes/"
    copy_file(path1, path2)