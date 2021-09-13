import os
import cv2
import numpy as np
import json
# from osgeo import gdal

def get_mvs_old():
    tif_path = "dataset/GID/Large-scale Classification_5classes"
    rgb_path = os.path.join(tif_path, "image_RGB")
    nrgb_path = os.path.join(tif_path, "image_NirRGB")
    label_path = os.path.join(tif_path, "label_5classes")
    tifs_rgb = os.listdir(rgb_path)
    tifs_nrgb = os.listdir(nrgb_path)
    tifs_label = os.listdir(label_path)
    b_list = []
    g_list = []
    r_list = []
    for i, tif in enumerate(tifs_rgb):
        img = cv2.imread(os.path.join(rgb_path, tif))    # bgr hwc
        # print(img.shape, img.min(), img.max())
        b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
        b_list.append(b.flatten())
        print(i, b.shape)
        g_list.append(g.flatten())
        r_list.append(r.flatten())
    B = np.concatenate(b_list)
    print(B.shape, B.size)
    G = np.concatenate(g_list)
    print(G.shape, G.size)
    R = np.concatenate(r_list)
    print(R.shape,R.size)

    numel = 6800*7200*150
    assert B.size == numel and G.size== numel and R.size==numel

    b_mean = np.mean(B)
    b_var = np.var(B)
    b_std = np.std(B)
    print(b_mean, b_var, b_std)

    g_mean = np.mean(G)
    g_var = np.var(G)
    g_std = np.std(G)

    r_mean = np.mean(R)
    r_var = np.var(R)
    r_std = np.std(R)


    str = "BGR平均值为:\n mean=[{}, {}, {}]\nBGR方差为:\nvar=[{}, {}, {}]\nBGR标准差为:\nstd=[{}, {}, {}]".\
        format(b_mean, g_mean, r_mean, b_var, g_var, r_var, b_std, g_std,r_std)
    print(str)
    fh = open('static.txt', 'w', encoding='utf-8')
    fh.write(str)
    fh.close()

def get_statics(tif_path, txt_path):
    rgb_path = os.path.join(tif_path, "image_RGB")
    rgb_path = os.path.join(tif_path, "train_set/rgb_bg")
    # nrgb_path = os.path.join(tif_path, "image_NirRGB")
    # label_path = os.path.join(tif_path, "label_5classes")
    tifs_rgb = os.listdir(rgb_path)
    # tifs_nrgb = os.listdir(nrgb_path)
    # tifs_label = os.listdir(label_path)

    b = 0
    g = 0
    r = 0
    for i, tif in enumerate(tifs_rgb):
        print(i, os.path.join(rgb_path, tif))
        img = cv2.imread(os.path.join(rgb_path, tif)) / 255.  # bgr hwc
        b += np.sum(img[:, :, 0])
        g += np.sum(img[:, :, 1])
        r += np.sum(img[:, :, 2])

    numel = 6800 * 7200 * 150  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
    b_mean = b / numel
    g_mean = g / numel
    r_mean = r / numel

    b = 0
    g = 0
    r = 0
    for i, tif in enumerate(tifs_rgb):
        print(i, os.path.join(rgb_path, tif))
        img = cv2.imread(os.path.join(rgb_path, tif)) / 255.  # bgr hwc
        b += np.sum((img[:, :, 0] - b_mean) ** 2)
        g += np.sum((img[:, :, 1] - g_mean) ** 2)
        r += np.sum((img[:, :, 2] - r_mean) ** 2)

    b_var = np.sqrt(b / numel)
    g_var = np.sqrt(g / numel)
    r_var = np.sqrt(r / numel)

    b_std = np.sqrt(b_var)
    g_std = np.sqrt(g_var)
    r_std = np.sqrt(r_var)

    str = "BGR平均值为:\nmean=[{}, {}, {}]\nBGR方差为:\nvar=[{}, {}, {}]\nBGR标准差为:\nstd=[{}, {}, {}]". \
        format(b_mean, g_mean, r_mean, b_var, g_var, r_var, b_std, g_std, r_std)
    print(str)
    fh = open(txt_path, 'w', encoding='utf-8')
    fh.write(str)
    fh.close()


def crop_tif(tif_path, RGB_label, crop_rgb_path, crop_rgb_bg_path, crop_label_color_path, crop_label_gray_path, size=224):
    # with open(RGB_label, "r") as f:  # 打开文件
    #     lines = f.readlines()  # 读取文件
    #     # print(lines)
    # lines = [line.split("\n")[0] for line in lines]
    # for i, l in enumerate(lines):
    #     print(i, l)


    rgb_path = os.path.join(tif_path, "image_RGB")
    # rgb_path = os.path.join("/home/liufang/project/DataSets/GID/Large-scale Classification_5classes", "image_RGB")
    # nrgb_path = os.path.join(tif_path, "image_NirRGB")

    label_path = os.path.join(tif_path, "label_5classes")
    # label_path = os.path.join("/home/liufang/project/DataSets/GID/Large-scale Classification_5classes", "label_5classes")
    # tifs_rgb = os.listdir(rgb_path)
    # tifs_nrgb = os.listdir(nrgb_path)
    # tifs_label = os.listdir(label_path)

    #==========================================================================
    h, w = 6800, 7200
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
    # for _, tif in enumerate(tifs_rgb):
    for _, tif in enumerate(lines):
        print(len(lines), tif)
        tif_rgb = os.path.join(rgb_path, tif)
        tif_label = os.path.join(label_path, tif.split(".")[0] + "_label.tif")
        print(tif_rgb)
        print(tif_label)

        img = cv2.imread(tif_rgb)
        label_color = cv2.imread(tif_label)
        label_gray = cv2.imread(tif_label, cv2.IMREAD_GRAYSCALE)
        assert img.shape == (6800, 7200, 3) and label_color.shape == (6800, 7200, 3) and label_gray.shape == (6800, 7200),\
            print(img.shape, label_gray.shape, label_color.shape)

        for i in range(lenght):
            print(i, h_cor[i], w_cor[i])
            tl_h, tl_w = h_cor[i], w_cor[i]
            print(tl_h, tl_h + size, tl_w, tl_w + size)

            crop_img = img[tl_h:tl_h + size, tl_w:tl_w + size, :]
            crop_label_gray= label_gray[tl_h:tl_h + size, tl_w:tl_w + size]
            crop_label_color = label_color[tl_h:tl_h + size, tl_w:tl_w + size, :]

            mask1 = crop_label_gray== 76
            mask2 = crop_label_gray== 150
            mask3 = crop_label_gray== 179
            mask4 = crop_label_gray== 226
            mask5 = crop_label_gray== 29
            mask = mask1+mask2+mask3+mask4+mask5
            mask = mask > 0

            if mask.sum() > size*size*0.1:
                mask = np.tile(mask[:,:, np.newaxis], 3)

                crop_imgbg = crop_img.copy()
                crop_imgbg = crop_imgbg * mask

                crop_label_gray = crop_label_gray * mask[:,:,0]
                crop_label_color = crop_label_color * mask

                saveimg = crop_rgb_path + tif.split(".")[0] + "_" + str(i) + ".tif"
                saveimgbg = crop_rgb_bg_path + tif.split(".")[0] + "_" + str(i) + ".tif"
                savelabelcolor = crop_label_color_path + tif.split(".")[0] + "_" + str(i) + ".tif"
                savelabelgray = crop_label_gray_path + tif.split(".")[0] + "_" + str(i) + ".tif"

                print(_, "/", i, count, tl_h,"-->",tl_h + size, tl_w, "-->", tl_w + size, saveimg)
                print(_, i, count, savelabelcolor, savelabelgray)

                cv2.imwrite(saveimg, crop_img)
                cv2.imwrite(saveimgbg, crop_imgbg)
                cv2.imwrite(savelabelcolor, crop_label_color)
                cv2.imwrite(savelabelgray, crop_label_gray)
                count += 1

def count(arr):
    from collections import Counter
    count = Counter(arr.flatten().tolist())
    return count



def make_5classes(crop_label_path, five_class_path):    # five_class_path
    label_path = os.listdir(crop_label_path)
    # assert len(label_path) == 144000, print(len(label_path))
    print(len(label_path))
    # classes = [76, 150, 179, 226, 29]
    #     classes_new = [1, 2, 3, 4, 5]
    classes = [76, 150, 179, 29]
    classes_new = [1, 2, 3, 4]
    count = 0

    for i, f in enumerate(label_path):
        p = os.path.join(crop_label_path, f)

        label = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        print(i, p, label.shape, label.min(), label.max())
        label_new = np.zeros_like(label)
        for j in range(len(classes)):
            label_new += ((label == classes[j]).astype(np.uint8) * classes_new[j])

        # new_path = os.path.join(five_class_path, f)
        new_path = os.path.join(five_class_path, f)
        print(i, new_path, label_new.shape, label_new.min(), label_new.max())
        cv2.imwrite(new_path, label_new)

        # cv2.imshow('old', label_color)
        # cv2.imshow("new", label_new)
        # cv2.waitKey(1)


        # if label_new.max() == 0:
        #     print("count", count)
        #     cv2.imshow('old', label_color)
        #     cv2.imshow("new", label_new)
        #     cv2.waitKey(1)
        #     count += 1

    # fh = open("no_zero.txt", 'w', encoding='utf-8')
    # for ii in no_zero:
    #     fh.write(json.dumps(ii) + '\n')
    # fh.close()



def split_dataset(rgb_path, label_path, train_txt, val_txt, test_txt):
    img_list = os.listdir(rgb_path)
    label_list = os.listdir(label_path)
    print(len(img_list), len(label_list))

    assert len(img_list) == len(label_list), print(len(img_list), len(label_list))
    num = max(len(img_list), len(label_list))

    for i in range(num):
        print(i, img_list[i], label_list[i])
        assert img_list[i] == label_list[i]

    train_list = []
    for i in range(num):
        train_dic = {}
        assert img_list[i] == label_list[i], print(i, img_list[i], label_list[i])
        train_dic["fpath_img"] = os.path.join("train_set/rgb_bg/", img_list[i])
        train_dic["fpath_segm"] = os.path.join("train_set/label_cls/", label_list[i])
        train_dic["width"] = 256
        train_dic["height"] = 256
        train_list.append(train_dic)
    #
    # val_list = []
    # for i in range(60000, 80000):
    #     val_dic = {}
    #     assert img_list[i] == label_list[i], print(i, img_list[i], label_list[i])
    #     val_dic["fpath_img"] = os.path.join("train_set/rgb_bg/",img_list[i])
    #     val_dic["fpath_segm"] = os.path.join("train_set/label5classes/", label_list[i])
    #     val_dic["width"] = 56
    #     val_dic["height"] = 56
    #     val_list.append(val_dic)
    #
    # test_list = []
    # for i in range(80000, num):
    #     test_dic = {}
    #     assert img_list[i] == label_list[i], print(i, img_list[i], label_list[i])
    #     test_dic["fpath_img"] = os.path.join("train_set/rgb_bg/", img_list[i])
    #     test_dic["fpath_segm"] = os.path.join("train_set/label5classes/", label_list[i])
    #     test_dic["width"] = 56
    #     test_dic["height"] = 56
    #     test_list.append(test_dic)
    #
    fh = open(train_txt, 'w', encoding='utf-8')
    for line in train_list:
        fh.write(json.dumps(line) + '\n')
    fh.close()
    #
    # fh = open(val_txt, 'w', encoding='utf-8')
    # for line in val_list:
    #     fh.write(json.dumps(line) + '\n')
    # fh.close()
    #
    # fh = open(test_txt, 'w', encoding='utf-8')
    # for line in test_list:
    #     fh.write(json.dumps(line) + '\n')
    # fh.close()





    # {"fpath_img": "ADEChallengeData2016/images/training/ADE_train_00000001.jpg",
    #  "fpath_segm": "ADEChallengeData2016/annotations/training/ADE_train_00000001.png", "width": 683, "height": 512}


def show_statics(tif_path, txt_path):
    rgb_path = os.path.join(tif_path, "image_RGB")
    label_path = os.path.join(tif_path, "label_15classes")

    tifs_rgb = os.listdir(rgb_path)
    tifs_label = os.listdir(label_path)

    print(os.path.join(rgb_path, tifs_rgb[0]))
    print(os.path.join(label_path, tifs_label[0]))
    img = cv2.imread(os.path.join(rgb_path, tifs_rgb[0]))
    label = cv2.imread(os.path.join(label_path, tifs_label[0]))
    print(img.shape, label.shape)

    from collections import Counter
    # result = Counter(list(img.flatten()))
    # print("result", result)

    b, g, r = label[:, :, 0], label[:, :, 1], label[:, :, 2]
    mask_1 = (r == 0)
    mask_2 = (r == 150)
    mask_3 = (r == 200)
    mask_4 = (r == 250)
    mask_r = mask_1 + mask_2 + mask_3 + mask_4

    mask_1 = (g == 0)
    mask_2 = (g == 150)
    mask_3 = (g == 200)
    mask_4 = (g == 250)
    mask_g = mask_1 + mask_2 + mask_3 + mask_4

    mask_1 = (b == 0)
    mask_2 = (b == 150)
    mask_3 = (b == 200)
    mask_4 = (b == 250)
    mask_b = mask_1 + mask_2 + mask_3 + mask_4

    mask_bg = ((((b == 0) + (g == 0) + (r == 0)).astype(np.uint8) -1)*(-1)).astype(np.uint8)


    new_label = np.zeros_like(label)
    new_label[:, :, 0] = label[:, :, 0] * mask_r
    new_label[:, :, 1] = label[:, :, 1] * mask_g
    new_label[:, :, 2] = label[:, :, 2] * mask_b



    # new_img = new_img * np.tile(mask_bg[:,:,np.newaxis],(3))

    # colors = np.array([[200, 0, 0], [250, 0, 150], [200, 150, 150], [250, 150, 150], [0, 200, 0],
    #           [150, 250, 0], [150, 200, 150], [200, 0, 200], [150, 0, 250],[150, 150, 250],
    #           [250, 200, 0], [200, 200, 0],[0, 0, 200],[0, 150, 200],[0, 200, 250]])
    # H, W, _ = img.shape
    # for h in range(H):
    #     for w in range(W):
    #         # print("img[h, w, :]", img[h, w, :])
    #         for color in colors:
    #             # print("color", color)
    #             if (img[h, w, :] == color).all():
    #                 # print("==")
    #                 new_img[h, w,:] = img[h, w, :]
    #             # else:
    #             #     print("!=")

    label_gray = cv2.cvtColor(new_label, cv2.COLOR_BGR2GRAY)

    mask_bg = (label_gray != 0)
    new_img = np.zeros_like(img)
    new_img[:, :, 0] = img[:, :, 0] * mask_bg
    new_img[:, :, 1] = img[:, :, 1] * mask_bg
    new_img[:, :, 2] = img[:, :, 2] * mask_bg



    result = Counter(list(label_gray.flatten()))
    print("result", result)



    cls = [92, 117, 192, 60, 111, 180, 165, 23, 146, 83, 73, 161]
    value = [1,2,3,4,5,6,7,8,9,10,11,12]
    label_cls = np.zeros_like(label[:, :, 0])
    for index, c in enumerate(cls):
        label_cls += (label_gray == c).astype(np.uint8) * value[index]



    cv2.imwrite("label_color.tif", new_label)
    cv2.imwrite("label_gray.tif", label_gray)
    cv2.imwrite("label_cls.tif", label_cls)
    cv2.imwrite("img_color.tif", new_img)

    # cv2.imshow("img", img)
    # cv2.imshow("color", new_label)
    # cv2.imshow("gray", label_gray)
    # cv2.imshow("cls", label_cls)
    # cv2.waitKey(0)



    # b = 0
    # g = 0
    # r = 0
    # for i, tif in enumerate(tifs_rgb):
    #     print(i, os.path.join(rgb_path, tif))
    #     img = cv2.imread(os.path.join(rgb_path, tif)) / 255.  # bgr hwc
    #     b += np.sum(img[:, :, 0])
    #     g += np.sum(img[:, :, 1])
    #     r += np.sum(img[:, :, 2])
    #
    # numel = 6800 * 7200 * 150  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
    # b_mean = b / numel
    # g_mean = g / numel
    # r_mean = r / numel
    #
    # b = 0
    # g = 0
    # r = 0
    # for i, tif in enumerate(tifs_rgb):
    #     print(i, os.path.join(rgb_path, tif))
    #     img = cv2.imread(os.path.join(rgb_path, tif)) / 255.  # bgr hwc
    #     b += np.sum((img[:, :, 0] - b_mean) ** 2)
    #     g += np.sum((img[:, :, 1] - g_mean) ** 2)
    #     r += np.sum((img[:, :, 2] - r_mean) ** 2)
    #
    # b_var = np.sqrt(b / numel)
    # g_var = np.sqrt(g / numel)
    # r_var = np.sqrt(r / numel)
    #
    # b_std = np.sqrt(b_var)
    # g_std = np.sqrt(g_var)
    # r_std = np.sqrt(r_var)
    #
    # str = "BGR平均值为:\nmean=[{}, {}, {}]\nBGR方差为:\nvar=[{}, {}, {}]\nBGR标准差为:\nstd=[{}, {}, {}]". \
    #     format(b_mean, g_mean, r_mean, b_var, g_var, r_var, b_std, g_std, r_std)
    # print(str)
    # fh = open(txt_path, 'w', encoding='utf-8')
    # fh.write(str)
    # fh.close()

# if __name__ == '__main__':
#     # get mean, var, std of each channel
#     tif_path = "../dataset/CROP_GID_56/Large-scale Classification_5classes"    # /home/liufang/project/DataSets
#     txt_path = 'static_5lcasses.txt'
#     # get_statics(tif_path, txt_path)
#
#     # get crop img and label to assumed size
#     RGB_label = "/home/liufang/project/DataSets/CROP_GID_56/Large-scale Classification_5classes/train_set/RGB_label.txt"
#     crop_rgb_path = "/home/liufang/project/DataSets/CROP_GID_56/Large-scale Classification_5classes/train_set/rgb/"
#     crop_rgb_bg_path = "/home/liufang/project/DataSets/CROP_GID_56/Large-scale Classification_5classes/train_set/rgb_bg/"
#     crop_label_path_color = "/home/liufang/project/DataSets/CROP_GID_56/Large-scale Classification_5classes/train_set/label_color/"
#     crop_label_path_gray = "/home/liufang/project/DataSets/CROP_GID_56/Large-scale Classification_5classes/train_set/label_gray/"
#     # crop_tif(tif_path, RGB_label, crop_rgb_path, crop_rgb_bg_path, crop_label_path_color, crop_label_path_gray, 56)
#
#     # make gray value to 0-->classes
#     five_class_path = "/home/liufang/project/DataSets/CROP_GID_56/Large-scale Classification_5classes/train_set/label5classes/"
#     # make_5classes(crop_label_path_gray, five_class_path)
#
#     # get set_list txt
#     train_txt = "../dataset/CROP_GID_256/Fine Land-cover Classification_15classes/train_.txt"
#     val_txt = "../dataset/CROP_GID_256/Fine Land-cover Classification_15classes/val_.txt"
#     test_txt = "../dataset/CROP_GID_256/Fine Land-cover Classification_15classes/test_.txt"
#     crop_rgb_path = "/home/liufang/project/DataSets/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/rgb"
#     crop_cls_path = "/home/liufang/project/DataSets/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/cls"
#     # split_dataset(crop_rgb_path, crop_cls_path, train_txt, val_txt, test_txt)
#
#
#     tif_path = "../dataset/CROP_GID_256/Fine Land-cover Classification_15classes/"
#     txt_path = '../dataset/CROP_GID_256/Fine Land-cover Classification_15classes/static_15lcasses.txt'
#     # show_statics(tif_path, txt_path)
#
#     # get_statics("img_color.tif", "static_15lcasses.txt")


    # b = 0
    # g = 0
    # r = 0
    # img = cv2.imread("img_color.tif") / 255.  # bgr hwc
    # b = np.sum(img[:, :, 0])
    # g = np.sum(img[:, :, 1])
    # r = np.sum(img[:, :, 2])
    #
    # numel = 6800 * 7200
    # b_mean = b / numel
    # g_mean = g / numel
    # r_mean = r / numel
    #
    # b = 0
    # g = 0
    # r = 0
    #
    # img = cv2.imread("img_color.tif") / 255.  # bgr hwc
    # b += np.sum((img[:, :, 0] - b_mean) ** 2)
    # g += np.sum((img[:, :, 1] - g_mean) ** 2)
    # r += np.sum((img[:, :, 2] - r_mean) ** 2)
    #
    # b_var = np.sqrt(b / numel)
    # g_var = np.sqrt(g / numel)
    # r_var = np.sqrt(r / numel)
    #
    # b_std = np.sqrt(b_var)
    # g_std = np.sqrt(g_var)
    # r_std = np.sqrt(r_var)
    #
    # str = "BGR平均值为:\nmean=[{}, {}, {}]\nBGR方差为:\nvar=[{}, {}, {}]\nBGR标准差为:\nstd=[{}, {}, {}]". \
    #     format(b_mean, g_mean, r_mean, b_var, g_var, r_var, b_std, g_std, r_std)
    # print(str)
    # fh = open(txt_path, 'w', encoding='utf-8')
    # fh.write(str)
    # fh.close()




















    # size = 256
    # # ==========================================================================
    # h, w = 6800, 7200
    # x = np.arange(0, w, 256)
    # y = np.arange(0, h, 256)
    # print(len(x), x, x[:-1])
    # print(len(y), y, y[:-1])
    # grid = np.meshgrid(x[:-1], y[:-1])
    # print(len(grid), grid[0].shape, grid[1].shape)
    # w_cor = list(grid[0].flatten())
    # h_cor = list(grid[1].flatten())
    # print(len(w_cor), w_cor)
    # print(len(h_cor), h_cor)
    # assert len(w_cor) == len(h_cor), print(len(w_cor), len(h_cor))
    # lenght = len(w_cor)
    # # ============================================================================
    #
    # count = 0
    # img = cv2.imread("img_color.tif")
    # label_color = cv2.imread("label_color.tif")
    # label_cls = cv2.imread("label_cls.tif", cv2.IMREAD_GRAYSCALE)
    #
    #
    # assert img.shape == (6800, 7200, 3) and label_color.shape == (6800, 7200, 3) and label_cls.shape == (6800, 7200), \
    #     print(img.shape, label_color.shape, label_cls.shape)
    #
    # for i in range(lenght):
    #     print(i, h_cor[i], w_cor[i])
    #     tl_h, tl_w = h_cor[i], w_cor[i]
    #     print(tl_h, tl_h + size, tl_w, tl_w + size)
    #
    #     crop_img = img[tl_h:tl_h + size, tl_w:tl_w + size, :]
    #     crop_label_cls = label_cls[tl_h:tl_h + size, tl_w:tl_w + size]
    #     crop_label_color = label_color[tl_h:tl_h + size, tl_w:tl_w + size, :]
    #
    #     # GF2_PMS2__L1A0001517494-MSS2.tif
    #     name = "GF2_PMS2__L1A0001517494-MSS2.tif"
    #     saveimgbg = "/home/liufang/project/DataSets/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/rgb_bg/" + name.split(".")[0] + "_" + str(i) + ".tif"
    #     savelabelcolor = "/home/liufang/project/DataSets/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/label_color/" + name.split(".")[0] + "_" + str(i) + ".tif"
    #     savelabelcls = "/home/liufang/project/DataSets/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/label_cls/" + name.split(".")[0] + "_" + str(i) + ".tif"
    #
    #     print(i, count, tl_h, "-->", tl_h + size, tl_w, "-->", tl_w + size, saveimgbg)
    #     print(count, savelabelcolor, savelabelcls)
    #
    #     cv2.imwrite(saveimgbg, crop_img)
    #     cv2.imwrite(savelabelcolor, crop_label_color)
    #     cv2.imwrite(savelabelcls, crop_label_cls)
    #     count += 1

    # size = 256
    # # ==========================================================================
    # h, w = 6800, 7200
    # x = np.arange(0, w, 256)
    # y = np.arange(0, h, 256)
    # print(len(x), x, x[:-1])
    # print(len(y), y, y[:-1])
    # grid = np.meshgrid(x[:-1], y[:-1])
    # print(len(grid), grid[0].shape, grid[1].shape)
    # w_cor = list(grid[0].flatten())
    # h_cor = list(grid[1].flatten())
    # print(len(w_cor), w_cor)
    # print(len(h_cor), h_cor)
    # assert len(w_cor) == len(h_cor), print(len(w_cor), len(h_cor))
    # lenght = len(w_cor)
    # # ============================================================================
    #
    # count = 0
    # img = cv2.imread("/home/liufang/project/DataSets/CROP_GID_256/Fine Land-cover Classification_15classes/image_RGB/GF2_PMS2__L1A0001517494-MSS2.tif")
    #
    #
    # assert img.shape == (6800, 7200, 3) ,print(img.shape)
    #
    # for i in range(lenght):
    #     print(i, h_cor[i], w_cor[i])
    #     tl_h, tl_w = h_cor[i], w_cor[i]
    #     print(tl_h, tl_h + size, tl_w, tl_w + size)
    #
    #     crop_img = img[tl_h:tl_h + size, tl_w:tl_w + size, :]
    #
    #     # GF2_PMS2__L1A0001517494-MSS2.tif
    #     name = "GF2_PMS2__L1A0001517494-MSS2.tif"
    #     saveimg = "/home/liufang/project/DataSets/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/img/" + \
    #                 name.split(".")[0] + "_" + str(i) + ".tif"
    #
    #
    #     print(i, count, tl_h, "-->", tl_h + size, tl_w, "-->", tl_w + size, saveimg)
    #     cv2.imwrite(saveimg, crop_img)
    #     count += 1
def filp():
    root = "/home/liufang/project/gidseg/dataset/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/"
    img_names = os.listdir(root + "rgb_bg/")
    label_color_name = os.listdir(root + "label_color/")
    label_cls_name = os.listdir(root + "label_cls/")

    assert len(img_names) == len(label_color_name) and len(img_names) == len(label_cls_name)
    length = len(img_names)
    for i in range(length):
        assert img_names[i] == label_color_name[i] and label_color_name[i] == label_color_name[i]
        name = img_names[i]

        img = cv2.imread(root + "rgb_bg/" + name)
        label_color = cv2.imread(root + "label_color/" + name)
        label_cls = cv2.imread(root + "label_cls/" + name, cv2.IMREAD_GRAYSCALE)
        print(img.shape)

        img1 = cv2.flip(img, 1)  # 图像水平翻转
        color1 = cv2.flip(label_color, 1)
        cls1 = cv2.flip(label_cls, 1)
        img1_path = root + "rgb/" + name.split(".")[0] + "_h.tif"
        color1_path = root + "color/" + name.split(".")[0] + "_h.tif"
        cls1_path = root + "cls/" + name.split(".")[0] + "_h.tif"
        cv2.imwrite(img1_path, img1)
        cv2.imwrite(color1_path, color1)
        cv2.imwrite(cls1_path, cls1)

        img0 = cv2.flip(img, 0)  # 图像垂直翻转
        color0 = cv2.flip(label_color, 0)
        cls0 = cv2.flip(label_cls, 0)
        img0_path = root + "rgb/" + name.split(".")[0] + "_v.tif"
        color0_path = root + "color/" + name.split(".")[0] + "_v.tif"
        cls0_path = root + "cls/" + name.split(".")[0] + "_v.tif"
        cv2.imwrite(img0_path, img0)
        cv2.imwrite(color0_path, color0)
        cv2.imwrite(cls0_path, cls0)

        img2 = cv2.flip(img, -1)  # 图像水平垂直翻转
        color2 = cv2.flip(label_color, -1)
        cls2 = cv2.flip(label_cls, -1)
        img2_path = root + "rgb/" + name.split(".")[0] + "_hv.tif"
        color2_path = root + "color/" + name.split(".")[0] + "_hv.tif"
        cls2_path = root + "cls/" + name.split(".")[0] + "_hv.tif"
        cv2.imwrite(img2_path, img2)
        cv2.imwrite(color2_path, color2)
        cv2.imwrite(cls2_path, cls2)

# root = "/home/liufang/project/gidseg/dataset/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/"
# root = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/"
# img_names = os.listdir(root + "rgb_bg/")
# label_color_name = os.listdir(root + "label_color/")
# label_cls_name = os.listdir(root + "label_cls/")
#
# assert len(img_names) == len(label_color_name) and len(img_names) == len(label_cls_name) and len(img_names) ==760
# length = len(img_names)
#
# for i in range(length):
#     assert img_names[i] == label_color_name[i] and label_color_name[i] == label_color_name[i]
#     name = img_names[i]
#
#     img = cv2.imread(root + "rgb_bg/" + name)
#     label_color = cv2.imread(root + "label_color/" + name)
#     label_cls = cv2.imread(root + "label_cls/" + name, cv2.IMREAD_GRAYSCALE)
#     print(img.shape)
#
#
#     img1 = cv2.flip(img, 1) #图像水平翻转
#     color1 = cv2.flip(label_color, 1)
#     cls1 = cv2.flip(label_cls, 1)
#     img1_path = root + "h_v_rgb/" + name.split(".")[0] + "_h.tif"
#     color1_path = root + "h_v_color/" + name.split(".")[0] + "_h.tif"
#     cls1_path = root + "h_v_cls/" + name.split(".")[0] + "_h.tif"
#     cv2.imwrite(img1_path, img1)
#     cv2.imwrite(color1_path, color1)
#     cv2.imwrite(cls1_path, cls1)
#
#
#
#     img0 = cv2.flip(img, 0)  # 图像垂直翻转
#     color0 = cv2.flip(label_color, 0)
#     cls0 = cv2.flip(label_cls, 0)
#     img0_path = root + "h_v_rgb/" + name.split(".")[0] + "_v.tif"
#     color0_path = root + "h_v_color/" + name.split(".")[0] + "_v.tif"
#     cls0_path = root + "h_v_cls/" + name.split(".")[0] + "_v.tif"
#     cv2.imwrite(img0_path, img0)
#     cv2.imwrite(color0_path, color0)
#     cv2.imwrite(cls0_path, cls0)
#
#
#
#     img2 = cv2.flip(img, -1)  # 图像水平垂直翻转
#     color2 = cv2.flip(label_color, -1)
#     cls2 = cv2.flip(label_cls, -1)
#     img2_path = root + "h_v_rgb/" + name.split(".")[0] + "_hv.tif"
#     color2_path = root + "h_v_color/" + name.split(".")[0] + "_hv.tif"
#     cls2_path = root + "h_v_cls/" + name.split(".")[0] + "_hv.tif"
#     cv2.imwrite(img2_path, img2)
#     cv2.imwrite(color2_path, color2)
#     cv2.imwrite(cls2_path, cls2)

def filp0():
    train_txt = "../dataset/CROP_GID_256/6classes/train_.txt"

    rgb_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/h_v_rgb"
    cls_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/h_v_cls"

    img_list = os.listdir(rgb_path)
    label_list = os.listdir(cls_path)
    print(len(img_list), len(label_list))

    assert len(img_list) == len(label_list), print(len(img_list), len(label_list))
    num = max(len(img_list), len(label_list))

    for i in range(num):
        print(i, img_list[i], label_list[i])
        assert img_list[i] == label_list[i]

    train_list = []
    for i in range(num):
        train_dic = {}
        assert img_list[i] == label_list[i], print(i, img_list[i], label_list[i])
        train_dic["fpath_img"] = os.path.join("train_set/h_v_rgb/", img_list[i])
        train_dic["fpath_segm"] = os.path.join("train_set/h_v_cls/", label_list[i])
        train_dic["width"] = 256
        train_dic["height"] = 256
        train_list.append(train_dic)

    fh = open(train_txt, 'w', encoding='utf-8')
    for line in train_list:
        fh.write(json.dumps(line) + '\n')
    fh.close()

# train_txt = "t.txt"
# rgb_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/extra_rgb"
# label_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/extra_cls"
# img_list = os.listdir(rgb_path)
# label_list = os.listdir(label_path)
# print(len(img_list), len(label_list))
#
# assert len(img_list) == len(label_list), print(len(img_list), len(label_list))
# num = max(len(img_list), len(label_list))
#
# for i in range(num):
#     print(i, img_list[i], label_list[i])
#     assert img_list[i] == label_list[i]
#
# train_list = []
# for i in range(num):
#     train_dic = {}
#     assert img_list[i] == label_list[i], print(i, img_list[i], label_list[i])
#     train_dic["fpath_img"] = os.path.join("train_set/extra_rgb/", img_list[i])
#     train_dic["fpath_segm"] = os.path.join("train_set/extra_cls/", label_list[i])
#     train_dic["width"] = 256
#     train_dic["height"] = 256
#     train_list.append(train_dic)
#
# fh = open(train_txt, 'w', encoding='utf-8')
# for line in train_list:
#     fh.write(json.dumps(line) + '\n')
# fh.close()

def xbq20():
    img = cv2.imread("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/xbq21/img.png")
    label_color = cv2.imread("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/xbq21/label.png")

    size = 256
    # ==========================================================================
    h, w,_ = img.shape
    x = np.arange(0, w, 256)
    y = np.arange(0, h, 256)
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
    colors = [0, 53 ,38, 15, 75, 267 ,113]    # 53, 38, 113, 75, ,
    print(img.shape, label_color.shape)

    for i in range(lenght):
        print(i, h_cor[i], w_cor[i])
        tl_h, tl_w = h_cor[i], w_cor[i]
        print(tl_h, tl_h + size, tl_w, tl_w + size)

        crop_img = img[tl_h:tl_h + size, tl_w:tl_w + size, :]
        crop_label_color = label_color[tl_h:tl_h + size, tl_w:tl_w + size, :]

        if crop_label_color.max() != 0 :

            gray = cv2.cvtColor(crop_label_color, cv2.COLOR_BGR2GRAY)
            # print("set", set(list(gray.flatten())))

            label_gray = np.zeros_like(gray)

            mask = (np.sum(crop_label_color, 2) != 0)
            if np.sum(mask.astype(np.uint8)) > 500:
                crop_img = crop_img * mask[:,:,np.newaxis]

                for ic, c in enumerate(colors):
                    m = (gray == c).astype(np.uint8) * ic
                    label_gray += m

                # cv2.imshow("img", crop_img)
                # cv2.imshow("mask", mask.astype(np.uint8)*255)
                # cv2.imshow("color", crop_label_color)
                # cv2.waitKey(0)

                saveimgbg = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/xbq21/rgb/" + "xbq_" + str(i) + ".png"
                savelabelcolor = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/xbq21/color/" + "xbq_" + str(i) + ".png"
                savelabelgray = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/xbq21/cls/" + "xbq_" + str(i) + ".png"

                print(i, count, tl_h, "-->", tl_h + size, tl_w, "-->", tl_w + size, saveimgbg)
                cv2.imwrite(saveimgbg, crop_img)
                cv2.imwrite(savelabelcolor, crop_label_color)
                cv2.imwrite(savelabelgray, label_gray)

        count += 1


def xbq20_txt():
    train_txt = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/xbq20/train_xbq20.txt"
    rgb_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/xbq20/rgb/"
    label_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/xbq20/cls/"
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
        train_dic["fpath_img"] = os.path.join("train_set/xbq21/rgb/", img_list[i])
        train_dic["fpath_segm"] = os.path.join("train_set/xbq21/cls/", label_list[i])
        train_dic["width"] = 256
        train_dic["height"] = 256
        train_list.append(train_dic)

    fh = open(train_txt, 'w', encoding='utf-8')
    for line in train_list:
        fh.write(json.dumps(line) + '\n')
    fh.close()

if __name__ == '__main__':
    xbq20_txt()