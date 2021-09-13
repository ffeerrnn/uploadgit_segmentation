class CONFIG():
    pic_num = -1    # -1 for all
    save_class = 2
    if_cuda = True
    # tif_path = "tifs/"    #
    # tif_path = "/home/liufang/project/DataSets/JSCZXB/data/geoserver/gwc/earth_R-CN-JSCZXB-202105-0_5M/EPSG_4326_17/426_173"
    # out_path = "result/"
    # result_classes = "result_classes/"

    # ===================================================
    # colors_path = "config/colors.txt"
    # colors_path = "config/colors_6.txt"
    colors_path = "config/colors_built.txt"
    test_h, test_w = 256, 256
    # ms1
    # mean = [0.011, 0.014, 0.013]    # RGB
    # std = [0.272, 0.294, 0.290]    # RGB

    # ms2
    # mean = [0.370, 0.390, 0.362]  # RGB
    # std = [0.493, 0.482, 0.483]    # RGB

    # ms3,0.5m-May  ms4,2m-April    best
    # mean = [0.192, 0.196, 0.177]
    # std = [0.491, 0.490, 0.481]
    #
    # ms5 2021shangbannian
    # mean = [0.285, 0.300, 0.247]  # RGB
    # std = [0.471, 0.479, 0.471]  # RGB

    #2019 shangbannian
    mean = [0.508, 0.507, 0.503]  # RGB
    std = [0.571, 0.569, 0.571]  # RGB

    # mean = [0.460, 0.464, 0.392]
    # std = [0.522, 0.526, 0.555]

    # BGR平均值为:
    # mean = [0.5032052455379737, 0.5076503988169089, 0.5082752353376714]
    # BGR方差为:
    # var = [0.3257004208465528, 0.3232563040615467, 0.32594803232174296]
    # BGR标准差为:
    # std = [0.5707016916450772, 0.5685563332349282, 0.570918586421692]


    layers = 50    # [18, 50]
    classes = 2    # 2, 7
    zoom_factor = 8
    # model_path = "exp/pspnet50/train_epoch_1000.pth"
    # model_path = "exp/pspnet50/train_epoch_187.pth"
    # model_path = "exp/pspnet50/train_epoch_727.pth"
    # model_path = "exp/pspnet50/train_epoch_201.pth"
    # model_path = "exp/pspnet50/train_epoch_226.pth"
    # model_path = "exp/pspnet50/train_epoch_240.pth"
    # model_path = "exp/pspnet50/train_epoch_371.pth"
    # model_path = "exp/pspnet50/train_epoch_200_built.pth"
    # model_path = "exp/pspnet50/train_epoch_200.pth"
    model_path = "exp/pspnet50/train_epoch_300_built.pth"
    # model_path = "exp/pspnet50/train_epoch_205.pth"
    # model_path = "exp/pspnet50/train_epoch_405.pth"
    # model_path = "exp/pspnet50/train_epoch_505.pth"
    # model_path = "exp/pspnet50/train_epoch_705.pth"
    # model_path = "exp/pspnet50/train_epoch_800.pth"
    # model_path = "exp/pspnet50/train_epoch_805.pth"
    # model_path = "exp/pspnet50/train_epoch_960.pth"
    # model_path = "exp/pspnet50/train_epoch_1160.pth"    # 0.5m-May    best
    # model_path = "exp/pspnet50/train_epoch_1360.pth"    # 2m
    # model_path = "exp/pspnet50/train_epoch_1660.pth"  # 2m
    # model_path = "exp/pspnet50/train_epoch_210.pth"  # 2m-April
    # model_path = "exp/pspnet50/train_epoch_1560.pth"  # 2m-April    best
    # model_path = "exp/pspnet50/train_epoch_1760.pth"  # 2m-April    best
    # model_path = "exp/pspnet50/train_epoch_1965.pth"
    # =====================================================

    batch_size = 1

    run_pass = True
    merge_classes = True

    # tif_path = "/home/liufang/project/DataSets/JSCZXB/xbq20/"
    # out_path = "/home/liufang/project/DataSets/JSCZXB/result/"
    # result_classes = "/home/liufang/project/DataSets/JSCZXB/result_classes/"

    # tif_path = "/home/liufang/project/DataSets/cz2m/R-CN-JSCZS-202104-2M-tiles/"
    # out_path = "/home/liufang/project/DataSets/cz2m/R-CN-JSCZS-202104-2M-tiles_result/"
    # result_classes = "/home/liufang/project/DataSets/cz2m/R-CN-JSCZS-202104-2M-tiles_result_classes/"

    # date = "202101"
    # tif_path = "/home/liufang/project/DataSets/cz2m/{}/R-CN-JSCZS-{}-2M-tiles/".format(date, date)
    # out_path = "/home/liufang/project/DataSets/cz2m/{}/R-CN-JSCZS-{}-2M-tiles_result/".format(date, date)
    # result_classes = "/home/liufang/project/DataSets/cz2m/{}/R-CN-JSCZS-{}-2M-tiles_result_classes/".format(date, date)

    # tif_path = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/R-CN-JSCZS-202106-tiles_4/"
    # out_path = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/R-CN-JSCZS-202106-tiles_built/"
    # result_classes = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/6R-CN-JSCZS-202106-tiles_result_classes/"

    tif_path = "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_0/"
    out_path = "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_built/"
    result_classes = "/home/liufang/Files/常州市影像图/2019年8月亚米图/R-CN-JSCZS-201908-0_8M-tiles_built_classes/"

    tif_path = "/home/liufang/project/demo/demo/test/"
    out_path = "/home/liufang/project/demo/demo/test_built/"
    result_classes = "/home/liufang/project/demo/demo/test_built_classes/"



    # tif_path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/00/"
    # out_path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/00_result/"
    # result_classes = "/home/liufang/Files/2021fh_sub_tifs/2_sub/00_result_classes/"
    #
    # tif_path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/22/"
    # out_path = "/home/liufang/Files/2021fh_sub_tifs/2_sub/22_result/"
    # result_classes = "/home/liufang/Files/2021fh_sub_tifs/2_sub/22_result_classes/"

    # tif_path = "/home/liufang/Files/tiff/0_4/"
    # out_path = "/home/liufang/Files/tiff/0_4_result/"
    # result_classes = "home/liufang/Files/tiff/0_4_result_classes/"

    byte_range = 700    # 1784 818    700


cfg = CONFIG()