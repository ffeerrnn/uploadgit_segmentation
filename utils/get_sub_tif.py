import cv2
import numpy as np
from skimage import io
# img = cv2.imread("/home/liufang/project/DataSets/cz2m/054528_022009.png")    #cz_202104_2m.tif
import os

def get_sub_tifs(path, save_path, size=256):

    if not os.path.exists(save_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(save_path)

    img = io.imread(path)
    # io.imsave("/home/liufang/project/DataSets/cz2m/cz_202104_2m.png", img)
    print("img.shape", img.shape)
    h, w, _ = img.shape
    if size:
        nh, nw = int(np.floor(h/size)), int(np.floor(w/size))
        img = img[:nh*size, :nw*size, :]
        print("img.shape", img.shape)
    else:
        nh = 3
        nw = 3
    h, w, _ = img.shape
    sh = int(np.floor(h/nh))
    sw = int(np.floor(w/nw))
    print(nh, nw, sh, sw)
    count = 0
    for i in range(nw):
        for j in range(nh):
            up = j*sh
            down = (j+1)*sh
            left = i*sw
            right = (i+1)*sw
            im = img[up:down, left:right, :]
            name = save_path + "{}_{}.tif".format(i, j)
            # name = save_path + "{}{}.tif".format(i,j)
            # name = save_path + "{}{}.png".format(i, j)

            io.imsave(name, im)

            print(count, up,"->", down, left, "->", right, name)
            # cv2.imwrite(name, im)
            count += 1

def stitch_sub_tifs(path, save_path, size=256):

    if not os.path.exists(save_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(save_path)

    names = os.listdir(path)
    names.sort()

    h_list = [int(i.split("_")[1].split(".")[0]) for i in names]
    w_list = [int(i.split("_")[0]) for i in names]
    w_list.sort()
    h_list.sort()
    # print(w_list)
    # print(h_list)
    max_w, max_h = max(w_list), max(h_list)
    print(max_w, max_h)

    w_img = []
    for i in range(max_w+1):
        h_img = []
        for j in range(max_h+1):
            nm = str(i) + "_" + str(j) + ".tif"
            # print("name", nm)
            img = cv2.imread(os.path.join(path, nm))
            # print("img.shape", img.shape)
            h_img.append(img)
        temp = np.concatenate(h_img, 0)
        print("temp.shape", temp.shape)
        w_img.append(temp)

    sa = list(np.arange(0, len(w_img), size))
    if len(w_img) not in sa:
        sa.append(len(w_img))
    # sa = [0, 30, 50, 80, 100, 130, 150, 180, 200, 230, 250, 286]
    for i in range(len(sa) - 1):
        result = np.concatenate(w_img[sa[i]:sa[i + 1]], 1)

        print(i, "result.shape", result.shape, type(result), result.dtype)
        cv2.imwrite(save_path + "temp_{}.png".format(i), result)



if __name__ == '__main__':
    path = "/home/liufang/project/DataSets/cz2m/cz_202104_2m.tif"
    save_path = "/home/liufang/project/DataSets/cz2m/"

    path = "/home/liufang/project/DataSets/cz2m/202105/R-CN-JSCZ-202105-2M.tif"
    save_path = "/home/liufang/project/DataSets/cz2m/202105/sub_tifs/"

    path = "/home/liufang/project/DataSets/cz2m/202103/R-CN-JSCZS-202103-2M.tif"
    save_path = "/home/liufang/project/DataSets/cz2m/202103/sub_tifs/"

    # path = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/R-CN-JSCZS-202106-0_8M.tif"
    # save_path = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/sub_tifs/"

    # path = "/home/liufang/Files/2021fh_sub_tifs/2.tif"
    # save_path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/"
    path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/00.tif"
    save_path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/00/"
    save_path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/00_result_classes/"
    new_path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/00_stitch/"
    # get_sub_tifs(path, save_path, size=256)
    # stitch_sub_tifs(save_path, new_path, size=91)

    path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/22.tif"
    save_path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/22/"
    # get_sub_tifs(path, save_path, size=256)
    save_path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/22_result_classes/"
    new_path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/22_stitch/"
    stitch_sub_tifs(save_path, new_path, size=91)