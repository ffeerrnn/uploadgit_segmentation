# System libs
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from util.dataset import ValDataset
from util import config
from util.utility import *


def validate(loader_val, model, criterion, cfg):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, input_target in enumerate(loader_val):
        data_time.update(time.time() - end)
        img_data = input_target['img_data'].cuda()
        seg_data = input_target['seg_label'].cuda()
        # print(i, img_data.shape, img_data.min(), img_data.max(), seg_data.shape, seg_data.min(), seg_data.max())
        output = model(img_data)
        # loss = criterion(output, seg_data)
        # loss = torch.mean(loss)
        output = output.max(1)[1]

        print(i, output.shape)


# ====================================================================================================
#         import cv2
#         for ii, mm in enumerate(output):
#             mean = np.array(cfg.mean).reshape(3, 1, 1)  # RGB
#             std = np.array(cfg.std).reshape(3, 1, 1)
#             im1 = img_data[ii,:,:,:].cpu().numpy()
#             im1 = im1 * std + mean
#             im1 = (im1.transpose((1, 2, 0)) * 255).astype(np.uint8)
#             print("im1", im1.shape)
#             gr1 = mm.cpu().numpy().astype(np.uint8)*255
#             gr1 = gr1[:,:,np.newaxis]
#             # print("gr1", gr1.shape)
#             gr1 = np.repeat(gr1, 3, axis=2)
#             print("gr1", gr1.shape)
#             im_gr = cv2.addWeighted(im1, 0.6, gr1, 0.3, 0)
#             pp = "/home/liufang/Files/2021fh_sub_tifs/n2/{}_{}.png".format(i, ii)
#             print(pp)
#
#             cv2.imwrite(pp, im_gr)
# =======================================================================================================

        intersection, union, target = intersectionAndUnionGPU(output, seg_data, cfg.classes, cfg.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        # loss_meter.update(loss.item(), img_data.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        # if (i + 1) % cfg.print_freq == 0:
        #     logger.info('Test: [{}/{}] '
        #                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
        #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
        #                 'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
        #                 'Accuracy {accuracy:.4f}.'.format(i + 1, len(loader_val),
        #                                                   data_time=data_time,
        #                                                   batch_time=batch_time,
        #                                                   loss_meter=loss_meter,
        #                                                   accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)    # for classes
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)    # for classes
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(cfg.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc

# [2021-09-07 08:39:26,763 INFO val.py line 57 108241] Val result: mIoU/mAcc/allAcc 0.6976/0.8206/0.8457.
# [2021-09-07 08:39:26,763 INFO val.py line 59 108241] Class_0 Result: iou/accuracy 0.8006/0.8828.
# [2021-09-07 08:39:26,763 INFO val.py line 59 108241] Class_1 Result: iou/accuracy 0.5946/0.7585.
# [2021-09-07 08:39:26,763 INFO val.py line 60 108241] <<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<

#n0
# [2021-09-07 11:15:34,470 INFO val.py line 99 124400] Val result: mIoU/mAcc/allAcc 0.7105/0.8222/0.8590.
# [2021-09-07 11:15:34,470 INFO val.py line 101 124400] Class_0 Result: iou/accuracy 0.8211/0.9086.
# [2021-09-07 11:15:34,470 INFO val.py line 101 124400] Class_1 Result: iou/accuracy 0.5999/0.7358.
# [2021-09-07 11:15:34,470 INFO val.py line 102 124400] <<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<

#n1  v
# [2021-09-07 11:38:17,778 INFO val.py line 99 126630] Val result: mIoU/mAcc/allAcc 0.6369/0.7663/0.8141.
# [2021-09-07 11:38:17,778 INFO val.py line 101 126630] Class_0 Result: iou/accuracy 0.7711/0.8787.
# [2021-09-07 11:38:17,778 INFO val.py line 101 126630] Class_1 Result: iou/accuracy 0.5028/0.6540.
# [2021-09-07 11:38:17,778 INFO val.py line 102 126630] <<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<

#n2   h
# [2021-09-07 11:49:26,854 INFO val.py line 99 128054] Val result: mIoU/mAcc/allAcc 0.6871/0.8056/0.8449.
# [2021-09-07 11:49:26,854 INFO val.py line 101 128054] Class_0 Result: iou/accuracy 0.8049/0.8979.
# [2021-09-07 11:49:26,854 INFO val.py line 101 128054] Class_1 Result: iou/accuracy 0.5693/0.7133.
# [2021-09-07 11:49:26,854 INFO val.py line 102 128054] <<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<



def main(cfg):
    if_cuda = cfg.if_cuda
    layers = cfg.layers
    classes = cfg.classes
    model_path = cfg.val_model_path
    zoom_factor = cfg.zoom_factor

    global logger
    logger = get_logger()

    # config
    check(cfg)
    logger.info(cfg)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^model
    from model.pspnet import PSPNet
    model = PSPNet(layers=layers, classes=classes, zoom_factor=zoom_factor, pretrained=False)

    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(cfg.classes))
    # logger.info(model)
    if if_cuda:
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')

    if if_cuda:
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

    val_data = ValDataset(cfg)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=cfg.val_batch_size,
                                             shuffle=False)

    loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, None, cfg)

    # logger.info('mIoU_val', mIoU_val)
    # logger.info('mAcc_val', mAcc_val)
    # logger.info('allAcc_val', allAcc_val)
    # logger.info('loss_val', loss_val)
    # logger.info('mIoU_val', mIoU_val)
    # logger.info('mAcc_val', mAcc_val)
    # logger.info('allAcc_val', allAcc_val)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("--config", default="config/pspnet50.yaml", metavar="FILE",help="path to config file", type=str)
    # parser.add_argument("--gpus", default="0", help="gpus to use, e.g. 0-3 or 0,1,2,3")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER, )
    args = parser.parse_args()

    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)

    main(cfg)
