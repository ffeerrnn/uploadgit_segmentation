# System libs
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

from util.dataset import TrainDataset,ValDataset
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
        loss = criterion(output, seg_data)
        loss = torch.mean(loss)
        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, seg_data, cfg.classes, cfg.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), img_data.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % cfg.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(loader_val),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

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


def train(model, criterion, optimizer, scheduler, loader_train, cfg):

    if cfg.evaluate:
        val_data = ValDataset(cfg)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=cfg.batch_size_val,
                                                 shuffle=False)

    # train & val
    for epoch in range(cfg.start_epoch, cfg.epochs):

        epoch_log = epoch + 1

        batch_time = AverageMeter()
        data_time = AverageMeter()
        main_loss_meter = AverageMeter()
        aux_loss_meter = AverageMeter()
        loss_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        model.train()
        end = time.time()
        max_iter = cfg.epochs * len(loader_train)
        for i, input_target in enumerate(loader_train):
            data_time.update(time.time() - end)

            img_data = input_target['img_data'].cuda()
            seg_data = input_target['seg_label'].cuda()

            # img_data, seg_data = Variable(img_data.cuda()), Variable(seg_data.cuda())
            # print(i, img_data.shape, img_data.min(), img_data.max(), seg_data.shape, seg_data.min(), seg_data.max())

            output, main_loss, aux_loss = model(img_data, seg_data)
            main_loss = torch.mean(main_loss)
            aux_loss = torch.mean(aux_loss)
            loss = main_loss + cfg.aux_weight * aux_loss

            optimizer.zero_grad()  # 梯度置零,因为反向传播过程中梯度会累加上一次循环的梯度
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新
            # scheduler.step()

            # for iter
            n = img_data.size(0)
            intersection, union, target = intersectionAndUnionGPU(output, seg_data, cfg.classes, cfg.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            main_loss_meter.update(main_loss.data.item(), n)
            aux_loss_meter.update(aux_loss.data.item(), n)
            loss_meter.update(loss.data.item(), n)
            batch_time.update(time.time() - end)
            end = time.time()

            current_iter = epoch * len(loader_train) + i + 1
            current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
            # print("len(optimizer.param_groups)", len(optimizer.param_groups))
            for index in range(0, 5):
                optimizer.param_groups[index]['lr'] = current_lr
            for index in range(5, len(optimizer.param_groups)):
                optimizer.param_groups[index]['lr'] = current_lr * 10



            remain_iter = max_iter - current_iter
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            # calculate accuracy, and display
            if (i + 1) % cfg.print_freq == 0:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'MainLoss {main_loss_meter.val:.4f} '
                            'AuxLoss {aux_loss_meter.val:.4f} '
                            'Loss {loss_meter.val:.4f} '
                            'Accuracy {accuracy:.4f}.'.format(epoch_log,
                                                              cfg.epochs,
                                                              i + 1,
                                                              len(loader_train),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              remain_time=remain_time,
                                                              main_loss_meter=main_loss_meter,
                                                              aux_loss_meter=aux_loss_meter,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))

            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

        # for epoch
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)    # each classes all in
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)    # for classes
        mIoU = np.mean(iou_class)    #
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        loss_train, mIoU_train, mAcc_train, allAcc_train = main_loss_meter.avg, mIoU, mAcc, allAcc
        # loss_train, mIoU_train, mAcc_train = main_loss_meter.avg, mIoU, mAcc
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.
                format(epoch_log, cfg.epochs, mIoU, mAcc, allAcc))
        writer.add_scalar('loss_train', loss_train, epoch_log)
        writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
        writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if epoch_log % cfg.save_freq == 0:
            filename = cfg.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       filename)
            if epoch_log / cfg.save_freq > 2:
                deletename = cfg.save_path + '/train_epoch_' + str(epoch_log - cfg.save_freq * 2) + '.pth'
                os.remove(deletename)

        # evaluation
        if cfg.evaluate and (epoch_log % cfg.val_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion, cfg)
            writer.add_scalar('loss_val', loss_val, epoch_log)
            writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
            writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
            writer.add_scalar('allAcc_val', allAcc_val, epoch_log)

    writer.close()




def main(cfg):
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.save_path)

    # config
    check(cfg)
    logger.info(cfg)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^model
    setup_seed(cfg.manual_seed)

    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label)    # ignore_label>classes-1,所有值都参与loss计算
    if cfg.arch == 'psp':
        from model.pspnet import PSPNet
        model = PSPNet(layers=cfg.layers,classes=cfg.classes,
                       zoom_factor=cfg.zoom_factor,criterion=criterion,pretrained=False)
    modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    modules_new = [model.ppm, model.cls, model.aux]

    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=cfg.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=cfg.base_lr * 10))

    # optimizer = torch.optim.SGD(model.parameters(),lr=cfg.base_lr,momentum=cfg.momentum,weight_decay=cfg.weight_decay)
    optimizer = torch.optim.SGD(params_list, lr=cfg.base_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=0.1, last_epoch=-1)
    scheduler = None

    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(cfg.classes))
    logger.info(model)
    model.cuda()
    if cfg.start_epoch: # 若非从头(0)训练,加载预训练模型
        resume_pth = cfg.save_path + "train_epoch_{}.pth".format(cfg.start_epoch)
        if os.path.isfile(resume_pth):
            logger.info("=> loading checkpoint '{}'".format(resume_pth))
            # checkpoint = torch.load(resume_pth, map_location=lambda storage, loc: storage.cuda())
            checkpoint = torch.load(resume_pth)
            assert cfg.start_epoch == checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # print("checkpoint['optimizer']", checkpoint['optimizer'])
            # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)    # optional
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_pth, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(resume_pth))
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv model

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^dataset
    dataset_train = TrainDataset(cfg)
    loader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,    # worker加载数据到RAM
                                               shuffle=True,
                                               # pin_memory=True,    # 在GPU上面预留一片固定的内存区域，以加速传输
                                               drop_last=True,)
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv dataset

    train(model, criterion, optimizer, scheduler, loader_train, cfg)


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
