import torch
import numpy as np



def get_one_hot_torch(label, N):
    """Gets one_hot.
        Args:
          label: Int torch, gray mode, [h,w], 0<=value<=class-1.
          N: class num.
        Returns:
          [h, w, cls] value is 0 or 1 , as net input.
        """
    size = (list(label.size()))    # [h,w]
    size.append(N)   # [h,w,cls]
    label = label.view(-1)   # 重构张量成一维
    ones = torch.sparse.torch.eye(N)    # 稀疏张量,对角线位置全1，其它位置全0
    ones = ones.index_select(0, label)   # 参数0表示按行索引1表示按列进行索引, label行号
    print("size",label.size(), size)
    return ones.view(*size)


# if __name__ == '__main__':
    # np.random.seed(1)
    # gt = np.random.randint(0,5, size=[6,6])  #先生成一个15*15的label，值在5以内，意思是5类分割任务
    # gt = torch.LongTensor(gt)
    # print("gt", gt.shape, gt)
    # gt_one_hot = get_one_hot_torch(gt, 5)


if __name__ == '__main__':
    import cv2
    img_name = "/home/liufang/project/semantic-segmentation/dataset/CROP_GID_256/" \
               "Fine Land-cover Classification_15classes/train_set/label_cls/GF2_PMS2__L1A0001517494-MSS2_4.tif"
    img = cv2.imread(img_name, 0)
    print(img.shape, img.min(), img.max())
    cv2.imshow("img", img)

    img = img.astype('uint8')
    # imgB = cv2.resize(imgB, (16, 16))
    print(img.shape, img, img.min(), img.max())

    img = torch.from_numpy(img).long()

    gt_one_hot = get_one_hot_torch(img, 7)

    print("gt_one_hot", gt_one_hot.shape)
    c0 = gt_one_hot[:, :, 0].numpy()
    c1 = gt_one_hot[:, :, 1].numpy()
    c2 = gt_one_hot[:, :, 2].numpy()
    c3 = gt_one_hot[:, :, 3].numpy()

    # print("bg", bg.shape, bg)
    cv2.imshow("c0", c0*255)
    cv2.imshow("c1", c1*255)
    cv2.imshow("c2", c2*255)
    cv2.imshow("c3", c3*255)

    cv2.waitKey(0)


