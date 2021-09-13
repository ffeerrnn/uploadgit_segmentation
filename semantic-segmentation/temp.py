import cv2
import os

def show_img():
    img = cv2.imread("/home/liufang/project/gidseg/dataset/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/rgb_bg/GF2_PMS2__L1A0001517494-MSS2_1.tif")
    label_color = cv2.imread("/home/liufang/project/gidseg/dataset/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/label_color/GF2_PMS2__L1A0001517494-MSS2_1.tif")
    label_cls = cv2.imread("/home/liufang/project/gidseg/dataset/CROP_GID_256/Fine Land-cover Classification_15classes/train_set/label_cls/GF2_PMS2__L1A0001517494-MSS2_1.tif", cv2.IMREAD_GRAYSCALE)

    cv2.imshow("img", img)
    cv2.imshow("color", label_color)
    cv2.imshow("cle", label_cls)
    cv2.waitKey(0)

def show_min_max():
    path = "/home/liufang/project/JSCZXB/data/geoserver/gwc/earth_R-CN-JSCZXB-202105-0_5M/EPSG_4326_17/426_173/"
    name = os.listdir(path)
    print(len(name))
    for i, n in enumerate(name):
        file = os.path.join(path, n)
        img = cv2.imread(file)
        print(i, n, img.min(), img.max())

def extra():
    import numpy as np
    name = "218442_088842.png"
    cls_name = "water"

    cls_names = ["bg", "water", "built", "unused", "farmlamd", "forest", "grassland"]
    # colors = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 0, 255], [0, 255, 0],[0, 255, 255],[255, 255, 0]]    # rgb
    colors = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255], [0, 255, 0],[255, 255, 0], [0, 255, 255]]    # bgr
    cls_nums = [0, 1, 2, 3, 4, 5, 6]

    index = cls_names.index(cls_name)
    color = colors[index]
    cls_num = cls_nums[index]
    print(name, index, cls_names[index], color, cls_num)

    path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/extra_rgb/"
    img = cv2.imread(os.path.join(path, name))

    mask = np.sum((img == 255), 2)
    mask = (mask != 3).astype(np.uint8)

    label_color = np.zeros_like(img)
    label_color = label_color + (np.array(color).reshape(1, 1, 3))
    label_color = label_color * mask[:, :, np.newaxis]
    label_color = label_color.astype(np.uint8)

    label_cls = np.zeros_like(mask)
    label_cls = mask * cls_num

    cv2.imwrite(os.path.join("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/extra_color", name), label_color)
    cv2.imwrite(os.path.join("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/extra_cls", name), label_cls)

    cv2.imshow("img", img)
    cv2.imshow("color", label_color)
    cv2.imshow("cls", label_cls)
    cv2.waitKey(0)



def _to_():
    # 2-->3
    import numpy as np
    path = "/home/liufang/project/semantic-segmentation/figure/demo/"
    name = "218330_088813.png"
    no, suf = name.split(".")
    img = cv2.imread(os.path.join(path, name))


    label_color = cv2.imread(os.path.join(path, no +"_color" + suf))    # "218330_088813_color.png"


    label_cls = cv2.imread(os.path.join(path, no +"_cls" + suf), cv2.IMREAD_GRAYSCALE)
    print(set(list(label_cls.flatten())))

    mask = (label_cls == 2).astype(np.uint8)
    un = np.zeros_like(img) + (np.array([255, 0, 255]).reshape(1, 1, 3))
    un = un * mask[:, :, np.newaxis]
    un = un.astype(np.uint8)
    print(un.shape)
    cv2.imshow("built", un)

    label_color_new = (label_color * (mask == 0)[:,:,np.newaxis]).astype(np.uint8)
    cv2.imshow("other", label_color_new)



    label_color_new += un
    label_color_new = label_color_new.astype(np.uint8)
    print(label_color_new.shape)

    coor = np.argwhere(label_cls == 2)
    label_cls[coor] = 3


    cv2.imwrite(os.path.join("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/extra_color", name), label_color_new)
    cv2.imwrite(os.path.join("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/extra_cls", name), label_cls)

    cv2.imshow("img", img)
    cv2.imshow("color", label_color_new)
    cv2.imshow("cls", label_cls)
    cv2.waitKey(0)


def extra_txt():
    import numpy as np
    path = "/home/liufang/project/semantic-segmentation/figure/demo/"
    name = "218452_088814.png"
    no, suf = name.split(".")
    img = cv2.imread(os.path.join(path, name))

    label_color = cv2.imread(os.path.join(path, no + "_color" + suf))  # "218330_088813_color.png"

    label_cls = cv2.imread(os.path.join(path, no + "_cls" + suf), cv2.IMREAD_GRAYSCALE)
    print(set(list(label_cls.flatten())))

    mask = (label_cls != 0).astype(np.uint8)
    un = np.zeros_like(img) + (np.array([255, 0, 255]).reshape(1, 1, 3))
    un = un * mask[:, :, np.newaxis]
    un = un.astype(np.uint8)
    print(un.shape)
    cv2.imshow("built", un)

    label_color_new = (label_color * (mask == 0)[:, :, np.newaxis]).astype(np.uint8)
    cv2.imshow("other", label_color_new)

    label_color_new += un
    label_color_new = label_color_new.astype(np.uint8)
    print(label_color_new.shape)

    coor = np.argwhere(label_cls == 2)
    label_cls[coor] = 3

    cv2.imwrite(os.path.join("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/extra_color", name),
                label_color_new)
    cv2.imwrite(os.path.join("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/extra_cls", name),
                label_cls)

    cv2.imshow("img", img)
    cv2.imshow("color", label_color_new)
    cv2.imshow("cls", label_cls)
    cv2.waitKey(0)












