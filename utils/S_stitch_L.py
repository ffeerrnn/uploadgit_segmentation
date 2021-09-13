import os
import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib
print(matplotlib.get_backend())

import skimage.io as io
import matplotlib.pyplot as plt

def s_stitch_l(path, savepath):

    if not os.path.exists(savepath):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(savepath)

    names = os.listdir(path)
    # name = "218258_088857.png"
    names.sort()

    wnum = []
    hnum = []

    current_w_nums = []
    current_h_nums = []
    listh = []
    for index, name in enumerate(names[1:]):  # 244*n+1

        w_num = name.split("_")[0]  # 列号
        h_num = name.split("_")[1].split(".")[0]  # 行号

        if index == 0:
            current_w_num = w_num
            current_h_num = h_num
        else:
            current_w_num = current_w_nums[-1]
            current_h_num = current_h_nums[-1]

        # print(index, name, current_w_nums, current_w_num, w_num)
        if index == 0:
            current_w_nums.append(w_num)
            current_h_nums.append(h_num)
        else:
            if current_w_num != w_num:
                current_w_nums.append(w_num)
                current_h_nums.append(h_num)

        file = os.path.join(path, name)
        # print(file)
        # img = io.imread(file)[:, :, :3].astype(np.uint8)
        # img = io.imread(file)
        img = cv2.imread(file)

        if index == 0:
            listw = [img]

        # print(index, name, current_w_nums, current_w_num, w_num)
        # print(index, name, current_w_num, w_num)
        if current_w_num == w_num:
            listw.append(img)
        else:
            listws = np.concatenate(listw[::-1], 0)
            print(index, listws.shape)
            # io.imshow(listws)
            # plt.show()

            listh.append(listws)

            listw = []
            listw.append(img)

        # wnum.append(w_num)
        # hnum.append(h_num)

    hh = []
    ww = []
    for index, i in enumerate(listh):
        print(index, i.shape)
        h, w, _ = i.shape
        # h, w = i.shape
        hh.append(h)
        ww.append(w)
    max_hh = max(hh)
    max_ww = max(ww)
    print(max_hh, max_ww)

    print("len(listh)", len(listh))

    # for index, i in enumerate(listh):
    #     print(i.shape, [3072-i.shape[0], 256, 3])
    #     if i.shape[0]<3072:
    #         i_new = np.concatenate((i,np.zeros([3072-i.shape[0], 256, 3], dtype=np.uint8)), 0)
    #         listh[index] = i_new

    # sa = [0, 30, 50, 80, 100, 130, 150, 180, 200, 230, 250, 286]
    sa = list(np.arange(0, len(listh), 90))
    if len(listh) not in sa:
        sa.append(len(listh))

    for i in range(len(sa) - 1):
        result = np.concatenate(listh[sa[i]:sa[i + 1]], 1)
        result = result.astype(np.uint8)
        print(i, "result.shape", result.shape, type(result), result.dtype)
        # io.imsave(savepath + "temp_color_{}.png".format(i), result)
        cv2.imwrite(savepath + "temp_color_{}.png".format(i), result)

        # io.imshow(result)

        # plt.show()

    # print(len(set(list(wnum))))
    # print(len(set(list(hnum))))
    #
    # #     file = os.path.join(path, name)
    # #
    # #     image = io.imread(file)
    # #     image = image
    # #     h.append(image)
    # #     print(count, name, image.shape)
    # #     count += 1
    # #
    # #
    # #     hs = np.concatenate(h, 0)
    # #     w.append(hs)
    # #
    # #     print(hs.shape)
    # #     io.imshow(hs)
    # #
    # #     plt.show()
    # #
    # # # hs = np.concatenate(h, 0)
    # # #
    # # # print(hs.shape)
    # # # io.imshow(hs)
    # # #
    # # # plt.show()


if __name__ == '__main__':
    # date = "202101"
    # path = "/home/liufang/project/DataSets/JSCZXB/data/geoserver/gwc/earth_R-CN-JSCZXB-202105-0_5M/EPSG_4326_17/426_173/"
    # path = "/home/liufang/project/DataSets/cz2m/{}/cat_R-CN-JSCZS-{}-2M-tiles_result_classes/".format(date, date)
    # savepath = "/home/liufang/project/DataSets/cz2m/{}/".format(date)

    # path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/00/"
    # savepath = "/home/liufang/Files/2021fh_sub_tifs/2_sub/00_stitch/"

    path = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/cat_R-CN-JSCZS-202106-tiles_result_classes/"
    savepath = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/"

    s_stitch_l(path, savepath)





