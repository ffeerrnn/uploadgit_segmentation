"from labelme img.png & label.png to dateset"

import cv2
import numpy as np
import os
import json

def xbq20(img_path, label_path, rgb_path, color_path, gray_path, colors, lab):

    if not os.path.exists(rgb_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(rgb_path)
    if not os.path.exists(color_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(color_path)
    if not os.path.exists(gray_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(gray_path)



    img = cv2.imread(img_path)
    label_color = cv2.imread(label_path)

    size = 256
    # ==========================================================================
    h, w,_ = img.shape
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
    # ============================================================================

    count = 0


    # colors = [0, 15, 38, 113, 75]
    # colors = [0, 53, 38, 113, 75, 15, ]
    # colors = [0, 53,38, 15, 75, 267 ,113]    #
    # colors = [0, 90, 113,  53, 15, 75, 38]
    # lab = [0,    1,    5,  3,  6,  4,  2]
    # colors = [0, 75, 38]
    # lab = [0, 1, 3]
    # colors = [0, 75]
    # lab = [0, 1]
    print(img.shape, label_color.shape)

    for i in range(lenght):
        print(i, h_cor[i], w_cor[i])
        tl_h, tl_w = h_cor[i], w_cor[i]
        print(tl_h, tl_h + size, tl_w, tl_w + size)

        crop_img = img[tl_h:tl_h + size, tl_w:tl_w + size, :]    # small img
        crop_label_color = label_color[tl_h:tl_h + size, tl_w:tl_w + size, :]    # small color

        if crop_label_color.max() != 0 :

            gray = cv2.cvtColor(crop_label_color, cv2.COLOR_BGR2GRAY)    # small label
            # print("set", set(list(gray.flatten())))

            label_gray = np.zeros_like(gray)

            mask = (np.sum(crop_label_color, 2) != 0)    # label mask
            if np.sum(mask.astype(np.uint8)) > 1000:    # size*size/6
                crop_img = crop_img * mask[:,:,np.newaxis]    # small img with mask

                for ic, c in enumerate(colors):
                    m = (gray == c).astype(np.uint8) * lab[ic]
                    label_gray += m    # small label cat

                # cv2.imshow("img", crop_img)
                # cv2.imshow("mask", mask.astype(np.uint8)*255)
                # cv2.imshow("color", crop_label_color)
                # cv2.waitKey(0)

                saveimgbg = rgb_path + str(i) + ".png"
                savelabelcolor = color_path + str(i) + ".png"
                savelabelgray = gray_path + str(i) + ".png"

                print(i, count, tl_h, "-->", tl_h + size, tl_w, "-->", tl_w + size, saveimgbg)
                cv2.imwrite(saveimgbg, crop_img)
                cv2.imwrite(savelabelcolor, crop_label_color)
                cv2.imwrite(savelabelgray, label_gray)

        count += 1

def xbq20_txt(train_txt, rgb_path, label_path, dest_rgb, dest_cls):

    img_list = os.listdir(rgb_path)
    label_list = os.listdir(label_path)
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
    img_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32/img.png"
    label_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32/label.png"

    rgb_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32/rgb/"
    color_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32/color/"
    gray_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32/cls/"

    # colors = [0, 38, 75, 15, 113, 53]
    # lab = [0, 2,4,3,6,1]
    colors = [0, 75, 38]
    lab = [0, 1, 5]

    xbq20(img_path, label_path, rgb_path, color_path, gray_path, colors, lab)

    # ===========================================================================================================

    train_txt = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32/cz2m_05_32.txt"
    rgb_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32/rgb/"
    label_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32/cls/"
    dest_rgb = "train_set/cz2m_05/32/rgb/"
    dest_cls = "train_set/cz2m_05/32/cls/"
    xbq20_txt(train_txt, rgb_path, label_path, dest_rgb, dest_cls)