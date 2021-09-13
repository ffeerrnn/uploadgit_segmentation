import cv2
import numpy as np
import os
import json


def from_5classes():
    in_str = ["GF2_PMS1__L1A0001094941-MSS1", "farmland_built", ]    # water, built, farmland, forest, meadow
    in_str = ["GF2_PMS1__L1A0001366278-MSS1", "water", ]
    in_str = ["GF2_PMS2__L1A0000564692-MSS2", "forest", ]
    in_str = ["GF2_PMS2__L1A0001573999-MSS2", "meadow",]
    in_str = ["GF2_PMS1__L1A0001015648-MSS1", "water"]
    in_str = ["GF2_PMS1__L1A0001491484-MSS1", "meadow"]
    in_str = ["GF2_PMS2__L1A0001378501-MSS2", "forest"]
    in_str = ["GF2_PMS2__L1A0001517494-MSS2", "built"]
    in_str = ["GF2_PMS1__L1A0000647767-MSS1", "built"]
    in_str = ["GF2_PMS1__L1A0001037899-MSS1", "built"]
    in_str = ["GF2_PMS1__L1A0001064469-MSS1", "built"]
    in_str = ["GF2_PMS1__L1A0001348919-MSS1", "built"]

    name = in_str[0]
    classes = in_str[1].split("_")

    img = cv2.imread("/home/liufang/project/DataSets/GID/Large-scale Classification_5classes/image_RGB/" + name + ".tif")
    col = cv2.imread("/home/liufang/project/DataSets/GID/Large-scale Classification_5classes/label_5classes/" + name + "_label.tif")
    gray = cv2.imread("/home/liufang/project/DataSets/GID/Large-scale Classification_5classes/label_5classes/" + name + "_label.tif",cv2.IMREAD_GRAYSCALE)

    # crop_rgb_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/"+in_str[1]+"/rgb/"
    # crop_col_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/"+in_str[1]+"/color/"
    # crop_cls_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/"+in_str[1]+"/cls/"
    # dest_rgb = "train_set/gf/" + in_str[1] + "/rgb/"
    # dest_cls = "train_set/gf/" + in_str[1] + "/cls/"
    # train_txt = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/" + in_str[1] + "/" + in_str[1] + ".txt"

    crop_rgb_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/test/" + in_str[1] + "/rgb/"
    crop_col_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/test/" + in_str[1] + "/color/"
    crop_cls_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/test/" + in_str[1] + "/cls/"
    dest_rgb = "train_set/test/"+in_str[1]+"/rgb/"
    dest_cls = "train_set/test/"+in_str[1]+"/cls/"
    train_txt = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/test/"+in_str[1]+"/"+in_str[1]+".txt"


    if not os.path.exists(crop_rgb_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(crop_rgb_path)
    if not os.path.exists(crop_col_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(crop_col_path)
    if not os.path.exists(crop_cls_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(crop_cls_path)

    print(set(list(gray.flatten())))


    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    m5 = 0
    m6 = 0
    if "water" in classes:
        m5 = (gray == 29).astype(np.uint8) * 1  # water
    if "built" in classes:
        m1 = (gray == 76).astype(np.uint8)*2    # built
    if "farmland" in classes:
        m2 = (gray == 150).astype(np.uint8)*4    # farmland
    if "forest" in classes:
        m3 = (gray == 179).astype(np.uint8)*5    # forest
    if "meadow" in classes:
        m4 = (gray == 226).astype(np.uint8)*6  # meadow

    cls = m1+m2+m3+m4+m5+m6

    mask = (cls != 0).astype(np.uint8)

    img = img * mask[:, :, np.newaxis]
    col = col * mask[:, :, np.newaxis]

    print(img.shape, col.shape)
    # cv2.imshow("img", img)
    # cv2.imshow("col", col)
    # cv2.imshow("mask", m1*255)
    # cv2.imshow("cls", cls)
    # cv2.waitKey(0)
    assert img.shape[0] == cls.shape[0] and img.shape[1] == cls.shape[1]

    size = 256
    # # ==========================================================================
    h, w = cls.shape
    x = np.arange(0, w, size)
    y = np.arange(0, h, size)
    print(len(x), x, x[:-1])
    print(len(y), y, y[:-1])
    grid = np.meshgrid(x[:-1], y[:-1])
    print(len(grid), grid[0].shape, grid[1].shape)
    w_cor = list(grid[0].flatten())
    h_cor = list(grid[1].flatten())
    print(len(w_cor), w_cor)
    print(len(h_cor), h_cor)
    assert len(w_cor) == len(h_cor), print(len(w_cor), len(h_cor))
    lenght = len(w_cor)
    # # ============================================================================
    count = 0
    for i in range(lenght):
        print(i, h_cor[i], w_cor[i])
        tl_h, tl_w = h_cor[i], w_cor[i]
        print(tl_h, tl_h + size, tl_w, tl_w + size)

        crop_img = img[tl_h:tl_h + size, tl_w:tl_w + size, :]
        crop_cls = cls[tl_h:tl_h + size, tl_w:tl_w + size]
        crop_col = col[tl_h:tl_h + size, tl_w:tl_w + size, :]

        mask = np.sum((crop_cls != 0).astype(np.uint8))

        if mask.sum() > size * size * 0.1:

            saveimg = crop_rgb_path + name + "_" + str(i) + ".tif"
            savelabelcolor = crop_col_path + name + "_" + str(i) + ".tif"
            savelabelgray = crop_cls_path + name + "_" + str(i) + ".tif"

            print(i, count, tl_h, "-->", tl_h + size, tl_w, "-->", tl_w + size, saveimg)
            print(i, count, savelabelcolor, savelabelgray)

            cv2.imwrite(saveimg, crop_img)
            cv2.imwrite(savelabelcolor, crop_col)
            cv2.imwrite(savelabelgray, crop_cls)
            count += 1


    #=============txt=============================================================
    img_list = os.listdir(crop_rgb_path)
    label_list = os.listdir(crop_cls_path)
    print(len(img_list), len(label_list))
    label_list = img_list

    assert len(img_list) == len(label_list), print(len(img_list), len(label_list))
    num = max(len(img_list), len(label_list))

    for i in range(num):
        print(i, img_list[i], label_list[i])
        assert img_list[i] == label_list[i]

    train_list = []
    for i in range(num):
        train_dic = {}
        assert img_list[i] == label_list[i], print(i, img_list[i], label_list[i])
        train_dic["fpath_img"] = os.path.join(dest_rgb, img_list[i])
        train_dic["fpath_segm"] = os.path.join(dest_cls, label_list[i])
        train_dic["width"] = 256
        train_dic["height"] = 256
        train_list.append(train_dic)

    fh = open(train_txt, 'w', encoding='utf-8')
    for line in train_list:
        fh.write(json.dumps(line) + '\n')
    fh.close()


if __name__ == '__main__':
    from_5classes()