import os
import cv2
import numpy as np

def get_statics(rgb_path, txt_path):
    tifs_rgb = []
    if isinstance(rgb_path, list):
        for rp in rgb_path:
            tifs = os.listdir(rp)
            tifs_rgb.extend([os.path.join(rp, tif) for tif in tifs])
    elif isinstance(rgb_path, str):
        tifs_rgb = os.listdir(rgb_path)

    b = 0
    g = 0
    r = 0
    numel = 0
    for i, tif in enumerate(tifs_rgb):
        print("mean:{}/{}".format(i, len(tifs_rgb)))
        img = cv2.imread(tif) / 255.  # bgr hwc
        h, w, _ = img.shape
        b += np.sum(img[:, :, 0])
        g += np.sum(img[:, :, 1])
        r += np.sum(img[:, :, 2])
        numel += h*w

    # numel = 6800 * 7200 * 150  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
    b_mean = b / numel
    g_mean = g / numel
    r_mean = r / numel
    print("BGR平均值为:\nmean=[{}, {}, {}]\n". format(b_mean, g_mean, r_mean))


    b = 0
    g = 0
    r = 0
    for i, tif in enumerate(tifs_rgb):
        print("var, std:{}/{}".format(i, len(tifs_rgb)))
        img = cv2.imread(tif) / 255.  # bgr hwc
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


if __name__ == '__main__':
    # rgb_path = ["/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf_for_01/farmland_built/rgb/",
    #             "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf_for_01/water/rgb/",
    #             "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf_for_01/meadow/rgb/",
    #             "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf_for_01/forest/rgb/"]
    # txt_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf_for_01/static.txt"

    # rgb_path = ["/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/farmland/rgb/",
    #             "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/built/rgb/",
    #             "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/water/rgb/",
    #             "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/meadow/rgb/",
    #             "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/forest/rgb/"]
    # txt_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/static.txt"

    # rgb_path = ["/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/farmland/rgb/",
    #             "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/built/rgb/",
    #             "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/water/rgb/",
    #             "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/meadow/rgb/",
    #             "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/forest/rgb/"]
    # txt_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/gf/static.txt"
    # get_statics(rgb_path, txt_path)

    rgb_path = ["/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_0/",
                 "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_1/",
                "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_2/",
                "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_3/",
                "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_4/",
                "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_5/"]
    txt_path = "/home/liufang/Files/常州市影像图/2019年8月亚米图/static_1.txt"
    get_statics(rgb_path, txt_path)