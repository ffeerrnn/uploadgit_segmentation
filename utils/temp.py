# import numpy as np
# color_RGB = np.array([[0, 0, 0],
#                       [0, 255, 0],
#                       [0, 0, 255],
#                       [255, 255, 0],
#                       [255, 0, 255],
#                       [0, 255, 255],
#                       [255, 0, 0]])
#
# print(color_RGB.shape)
# mask = (color_RGB.sum(1) > 0).astype(np.uint8) * 255
# mask = mask.reshape(-1, 1)
# print(mask.shape, mask)
#
# color_RGBA = np.concatenate((color_RGB, mask), 1)
# print(color_RGBA.shape, color_RGBA)

# import os
# path = "/media/liufang/KINGSTON/213_085"
# file_name = os.listdir(path)
# print(len(file_name))
# for name in file_name:
#     file_path = os.path.join(path, name)
#     f = open(file_path, "rb")
#     size = len(f.read())
#     # print(file_path, size)
#     if size <= 1784:  # byte  818    1784
#         os.remove(os.path.join(path, name))
#
# file_name = os.listdir(path)
# print(len(file_name))

import cv2
import numpy as np

import matplotlib.pyplot as plt

# global mean, std
# mean = [0.285, 0.300, 0.247]  # RGB
# std = [0.471, 0.479, 0.471]  # RGB

#2019 shangbannian
# mean = [0.508, 0.507, 0.503]  # RGB
# std = [0.571, 0.569, 0.571]  # RGB



# mean= 0.13392810454096926 ,std= 0.47099999451683844

def img_process(origin_img, mean = [0.285, 0.300, 0.247], std = [0.471, 0.479, 0.471]):

    # h, w, _ = origin_img.shape
    # origin_img = np.float32(np.array(origin_img)) / 255.
    # b = np.sum(origin_img[:, :, 0])
    # g = np.sum(origin_img[:, :, 1])
    # r = np.sum(origin_img[:, :, 2])
    # numel = h * w
    # b_mean = b / numel
    # g_mean = g / numel
    # r_mean = r / numel
    # mean = [r_mean, g_mean, b_mean]
    #
    # b = np.sum((origin_img[:, :, 0] - b_mean) ** 2)
    # g = np.sum((origin_img[:, :, 1] - g_mean) ** 2)
    # r = np.sum((origin_img[:, :, 2] - r_mean) ** 2)
    # b_var = np.sqrt(b / numel)
    # g_var = np.sqrt(g / numel)
    # r_var = np.sqrt(r / numel)
    # b_std = np.sqrt(b_var)
    # g_std = np.sqrt(g_var)
    # r_std = np.sqrt(r_var)
    # std = [r_std, g_std, b_std]
    # print(mean, std)
    # mean = [0.285, 0.300, 0.247]  # RGB
    # std = [0.471, 0.479, 0.471]  # RGB

    # mean = 0.2784571318200413, std = 0.47099999039580964
    # mean = 0.16127143402757965, std = 0.47099998512089297

    # mask = np.sum((origin_img == 255), 2)
    # mask = (mask != 3).astype(np.uint8)

    image = origin_img[:, :, ::-1]    # BGR->RGB
    # image = image * mask[:,:,np.newaxis]

    print("image", image.shape)
    im = image[:, :, 2-c]
    print("im.shape", im.shape)
    x1 = np.float(im[h1, w1])
    x2 = np.float(im[h2, w2])
    print("x1=", x1, ",x2=", x2)


    image = np.float32(np.array(image)) / 255.
    image = image.transpose((2, 0, 1))  # chw

    mean = np.array(mean).reshape(3, 1, 1)  # RGB
    std = np.array(std).reshape(3, 1, 1)

    image = (image - mean) / std

    image = image.transpose((1, 2, 0))  # chw

    print("image", image.shape)
    im = image[:, :, 2 - c]
    print("im.shape", im.shape)

    global y1, y2
    y1 = im[h1, w1]
    y2 = im[h2, w2]
    print("y1=", y1, ",y2=", y2)
    s = ((x1 - x2) / (y1 - y2)) / 255
    m = (x1 - y1 * s) / 255
    print("mean=", m, ",std=", s)

    return image



# r 2, g 1, b 0
c = 2
m_list = []
s_list = []
num = 10

for h1 in range(num):
    for w1 in range(num):
        for h2 in range(num):
            for w2 in range(num):
                img_source = cv2.imread("/home/liufang/project/demo/demo/test/217828_088317_r_2019.png")    # bgr
                print("img_source", img_source.shape)
                img_source = img_source[:, :, ::-1]
                im = img_source[:, :, c]
                print("im.shape", im.shape)
                x1 = im[h1, w1]
                x2 = im[h2, w2]
                print("x1=", x1, ",x2=", x2)
                img1 = img_process(img_source)


                # hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
                # plt.figure()
                # plt.title("Grayscale Histogram")
                # plt.xlabel("Bins")
                # plt.ylabel("number of Pixels")
                # plt.plot(hist)
                # plt.xlim([0, 256])
                # lab1 = cv2.imread("/home/liufang/project/demo/demo/test_built/217828_088317_r_2019_1.png")
                # re1 = cv2.addWeighted(img1, 0.5, lab1, 0.5, 0.1)

                # r 2, g 1, b 0

                img_dest = cv2.imread("/home/liufang/project/demo/demo/test/217828_088317_r_2021.png")
                print("img_dest", img_dest.shape)
                im = img_dest[:, :, c]
                print("im.shape", im.shape)
                x1_dest = np.float(im[h1, w1])
                x2_dest = np.float(im[h2, w2])
                print("\nx1=", x1_dest, ",x2=", x2_dest)
                print("y1=", y1, ",y2=", y2)
                s = ((x1_dest - x2_dest)/(y1 - y2))/255
                m = (x1 - y1 * s)/255
                print("mean=", m, ",std=", s)
                # if m:
                m_list.append(m)
                # if s:
                s_list.append(s)


x = np.arange(len(m_list))


# print("x", x)
# print("m_list", m_list)
# print("s_list", s_list)
plt.plot(x, m_list, marker='o', mec='r', mfc='w',label=u'm_list曲线图')
# plt.plot(x, s_list, marker='.', ms=10, label=u's_list曲线图')
plt.legend()  # 让图例生效

plt.margins(0)
plt.subplots_adjust(bottom=0.15)
m_list = [i for i in m_list if (i is not np.nan and i is not np.inf and i is not -np.inf)]
s_list = [i for i in s_list if i >= 0 or i < 0]
print("m_list", m_list)
print("s_list", s_list)
print("np.mean(m_list), np.mean(s_list)", np.mean(m_list), np.mean(s_list))

plt.show()









# img2 = img_process(img2)
# hist = cv2.calcHist([img2], [0], None, [256], [0, 256])
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("number of Pixels")
# plt.plot(hist)
# plt.xlim([0, 256])
# plt.show()
# lab2 = cv2.imread("/home/liufang/project/demo/demo/test_built/217828_088317_r_2021_1.png")
# re2 = cv2.addWeighted(img2, 0.5, lab2, 0.5, 0.1)
# cv2.imshow("217828_088317_r_2019", re1)
# cv2.imshow("217828_088317_r_2021_1", re2)
# cv2.waitKey(0)

