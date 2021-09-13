class CONFIG():
    pic_num = -1    # -1 for all
    save_class = 2
    if_cuda = True

    # ===================================================
    colors_path = "config/colors_built.txt"
    test_h, test_w = 256, 256

    mean = [0.285, 0.300, 0.247]  # RGB
    std = [0.471, 0.479, 0.471]  # RGB

    layers = 50    # [18, 50]
    classes = 2
    zoom_factor = 8
    model_path = "exp/pspnet50/train_epoch_200_built.pth"

    # =====================================================

    batch_size = 1

    run_pass = True
    merge_classes = True

    # txt_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/test/built/built_val.txt"
    # out_path = "/home/liufang/project/demo/demo/built/"
    # result_classes = "/home/liufang/project/demo/demo/built_classes/"

    txt_path = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/test/built/built_val.txt"
    out_path = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/sub_tifs/1_result/"
    result_classes = "/home/liufang/Files/常州市影像图/2021年上半年亚米图/sub_tifs/1_class/"

    byte_range = 700    # 1784 818    700


cfg = CONFIG()