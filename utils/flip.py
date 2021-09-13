import os
import json
import cv2

def flip(in_rgb, in_cls, out_rgb, out_cls, txt, dest_rgb, dest_cls):

    if not os.path.exists(out_rgb):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out_rgb)
    if not os.path.exists(out_cls):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(out_cls)

    img_names = os.listdir(in_rgb)
    cls_name = os.listdir(in_cls)

    assert len(img_names) == len(cls_name)
    length = len(img_names)

    for i in range(length):
        assert img_names[i] == cls_name[i]

        name = img_names[i]

        img = cv2.imread(os.path.join(in_rgb, name))
        label_cls = cv2.imread(os.path.join(in_cls, name), cv2.IMREAD_GRAYSCALE)
        print(img.shape)

        img0 = cv2.flip(img, 1)  # 图像水平翻转
        cls0 = cv2.flip(label_cls, 1)
        name0 = name.split(".")[0] + "_h." + name.split(".")[1]
        print(name, name0)
        cv2.imwrite(os.path.join(out_rgb, name0), img0)
        cv2.imwrite(os.path.join(out_cls, name0), cls0)

        img1 = cv2.flip(img, 0)  # 图像垂直翻转
        cls1 = cv2.flip(label_cls, 0)
        name1 = name.split(".")[0] + "_v." + name.split(".")[1]
        cv2.imwrite(os.path.join(out_rgb, name1), img1)
        cv2.imwrite(os.path.join(out_cls, name1), cls1)

        img2 = cv2.flip(img, -1)  # 图像水平垂直翻转
        cls2 = cv2.flip(label_cls, -1)
        name2 = name.split(".")[0] + "_hv." + name.split(".")[1]
        cv2.imwrite(os.path.join(out_rgb, name2), img2)
        cv2.imwrite(os.path.join(out_cls, name2), cls2)

    img_list = os.listdir(out_rgb)
    label_list = os.listdir(out_cls)
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

    fh = open(txt, 'w', encoding='utf-8')
    for line in train_list:
        fh.write(json.dumps(line) + '\n')
    fh.close()

if __name__ == '__main__':
    in_rgb = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/11/rgb/"
    in_cls = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/11/cls"
    out_rgb = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/11/rgb_flip/"
    out_cls = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/11/cls_flip"
    txt = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/11/cz2m_05_11_flip.txt"
    dest_rgb = "train_set/cz2m_05/11/rgb_flip/"
    dest_cls = "train_set/cz2m_05/11/cls_flip/"

    # "home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf_for_01/forest"
    in_rgb = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf_for_01/forest/rgb/"
    in_cls = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf_for_01/forest/cls"
    out_rgb = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf_for_01/forest/rgb_flip/"
    out_cls = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf_for_01/forest/cls_flip"
    txt = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf_for_01/forest/forest_flip.txt"
    dest_rgb = "train_set/gf_for_01/forest/rgb_flip/"
    dest_cls = "train_set/gf_for_01/forest/cls_flip/"
    flip(in_rgb, in_cls, out_rgb, out_cls, txt, dest_rgb, dest_cls)
