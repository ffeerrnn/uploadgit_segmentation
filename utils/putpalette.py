import numpy as np

color_RGB = np.array([[0, 0, 0],
                      [0, 255, 0],
                      [0, 0, 255],
                      [255, 255, 0],
                      [255, 0, 255],
                      [0, 255, 255],
                      [255, 0, 0]])

mask = (color_RGB.sum(1) > 0).astype(np.uint8) * 255
mask = mask.reshape(-1, 1)
color_RGBA = np.concatenate((color_RGB, mask), 1)
# color_RGBA = np.array([[0, 0, 0, 0],
#                   [0, 255, 0, 255],
#                   [0, 0, 255, 255],
#                   [255, 255, 0, 255],
#                   [255, 0, 255, 255],
#                   [0, 255, 255, 255],
#                   [255, 0, 0, 255]])


def colorize(img, color=color_RGB):
    """Gets colorized img.
        Args:
          img: Int torch, gray mode, [h,w], 0<=value<=class-1.
          color: RGB[[*,*,*]],shape is (n, 3) or RGBA[[*,*,*,*]], shape is (n, 4), n>=class.
        Returns:
          colorized img.
        """
    color_img = color[img].astype(np.uint8)
    return color_img


def colorize(gray, palette, transparent=False):
    from PIL import Image
    if not transparent:
        # gray: numpy array of the label and 1*3N size list palette
        color_img = Image.fromarray(gray.astype(np.uint8)).convert('P')
        color_img.putpalette(palette)

    else:
        #=========================================================================
        # color_img = Image.fromarray((gray==False).astype(np.uint8)).convert('P')
        # color_img.putpalette(palette)
        # color_img = color_img.convert("RGBA")
        # datas = color_img.getdata()
        # newData = []
        # for item in datas:
        #     if item[0] == 1 and item[1] == 1 and item[2] == 1:
        #         newData.append((0, 0, 0, 0))
        #     else:
        #         newData.append(item)
        # color_img.putdata(newData)
        #
        # datas = color_img.getdata()
        # for item in datas:
        #     print(item)
        # ============================================================================
        color_img = Image.fromarray((gray == False).astype(np.uint8)).convert('P')
        color_img.putpalette(palette)
        color_img = color_img.convert('RGB')
        color_img = np.array(color_img)
        mask = (color_img == 1)
        mask = np.sum(mask, 2)
        mask = (mask != 3)
        a = (mask * 255)[:,:,np.newaxis]
        color_img = color_img * mask[:, :, np.newaxis]
        color_img = np.concatenate((color_img, a), 2)
        color_img = Image.fromarray(color_img.astype(np.uint8)).convert("RGBA")
    return color_img

if __name__ == '__main__':
    import cv2
    img_name = "/home/liufang/project/semantic-segmentation/dataset/CROP_GID_256/" \
               "Fine Land-cover Classification_15classes/train_set/label_cls/GF2_PMS2__L1A0001517494-MSS2_4.tif"
    img = cv2.imread(img_name, 0)
    print(img.shape, img.min(), img.max())
    color_img = colorize(img)

    cv2.imshow("img", color_img)
    cv2.waitKey(0)
