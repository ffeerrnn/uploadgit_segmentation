import cv2
import numpy as np
import os
from skimage import io

def TifCrop(TifPath, SavePath, CropSize, stride):
    img = cv2.imread(TifPath)
    height, width, _ = img.shape
    print(img.shape, int((height-CropSize)/stride+1), int((width-CropSize)/stride+1))

    #  裁剪图片
    i_range, j_range = int((height-CropSize)/stride+1), int((width-CropSize)/stride+1)
    for i in range(i_range):    # 列
        for j in range(j_range):    # 行
            sh, eh = i*stride, i*stride+CropSize
            sw, ew = j*stride, j*stride+CropSize
            print(i, j, ":", sh,"-->", eh, sw, "-->", ew , "save to {}_{}.tif".format(i, i*i_range+j))
            cropped = img[sh: eh, sw: ew, :]
            # #  写图像
            # cv2.imwrite(SavePath + "{}_{}.tif".format(i, i*i_range+j), cropped)

def TifCom(TifsPath, SavePath, CropSize, stride):
    names = os.listdir(TifsPath)
    names.sort(key=lambda name:int(name.split("_")[1].split(".")[0]) )

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

        print(index, name,)
        if index == 0:
            current_w_nums.append(w_num)
            current_h_nums.append(h_num)
        else:
            if current_w_num != w_num:
                current_w_nums.append(w_num)
                current_h_nums.append(h_num)

        file = os.path.join(TifsPath, name)

        start = int((CropSize - stride)/2)
        img = cv2.imread(file)[start:start+stride, start:start+stride, :]
        print(file, img.shape, start, start+stride)
        # if img.ndim == 2:
        #     img = np.pad(img[:, :, np.newaxis], ((0, 0), (0, 0), (0, 3)))

        img = img[:, :, :3].astype(np.uint8)

        if index == 0:
            listw = [img]

        # print(index, name, current_w_nums, current_w_num, w_num)
        # print(index, name, current_w_num, w_num)
        if current_w_num == w_num:
            listw.append(img)
        else:
            listws = np.concatenate(listw, 1)
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
        hh.append(h)
        ww.append(w)
    max_hh = max(hh)
    max_ww = max(ww)
    print(max_hh, max_ww)


    result = np.concatenate(listh, 0)
    result = result.astype(np.uint8)
    print("result.shape", result.shape, type(result), result.dtype)
    cv2.imwrite(SavePath, result)


if __name__ == '__main__':

    # # 裁剪
    # TifCrop("/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/xbq20/xbq20.tif",
    #         "/home/liufang/project/DataSets/JSCZXB/xbq20/", 256, 128)

    # 拼接
    TifCom("/home/liufang/project/DataSets/JSCZXB/result_classes/",
           "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/xbq20/xbq20_color.png", 256, 128)

