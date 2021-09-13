import argparse
import base64
import json
import os
import os.path as osp

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils


def main(json_file, out, save_img_name, save_label_name, save_viz_name, save_txt_name):
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--json_file", default="/home/liufang/project/DataSets/cz2m/sub_cz2m/22.json")    # "/home/liufang/project/xbq20.json"
    # parser.add_argument("--json_file",
    #                     default="/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz33/33.json")  # "/home/liufang/project/xbq20.json"
    # parser.add_argument("--out", default="/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz33/")
    # args = parser.parse_args()
    # json_file = args.json_file
    # out = args.out

    if out is None:
        out_dir = osp.basename(json_file).replace(".", "_")
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = out

    print("out_dir", out_dir)
    if not osp.exists(out_dir):
        os.mkdir(out_dir)


    data = json.load(open(json_file))
    imageData = data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)    # 图像矩阵
    print("img", img.shape)

    label_name_to_value = {"_background_": 0}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(
        img.shape, data["shapes"], label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    print("label_names", label_names)
    lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
    )

    PIL.Image.fromarray(img).save(osp.join(out_dir, save_img_name))
    utils.lblsave(osp.join(out_dir, save_label_name), lbl)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, save_viz_name))

    with open(osp.join(out_dir, save_txt_name), "w") as f:
        for lbl_name in label_names:
            f.write(lbl_name + "\n")

    logger.info("Saved to: {}".format(out_dir))


if __name__ == "__main__":
    # json_file = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz12/12_1.json"
    # out = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz12/"

    # json_file = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32/32.json"
    # out = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32"
    # save_img_name = "img.png"
    # save_label_name = "label.png"
    # save_viz_name = "label_viz.png"
    # save_txt_name = "label_names.txt"
    #
    # main(json_file, out, save_img_name, save_label_name, save_viz_name, save_txt_name)

    # json_file = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32/32.json"
    # out = "/home/liufang/project/DataSets/CROP_GID_256/6classes/train_set/cz2m_05/32"
    # save_img_name = "img.png"
    # save_label_name = "label.png"
    # save_viz_name = "label_viz.png"
    # save_txt_name = "label_names.txt"
    import os
    path = "/home/liufang/Files/2021fh_sub_tifs/3_4_json/"
    names = os.listdir(path)
    # print(names)
    names = [ i.split(".")[0] for i in names]
    for name in names:
        json_file = os.path.join(path, name) + ".json"
        out = "/home/liufang/Files/2021fh_sub_tifs/3_4_samples/"
        save_img_name = name + "_img.png"
        save_label_name = name + "_corlor.png"
        save_viz_name = name + "_viz.png"
        save_txt_name = name + "_names.txt"


        main(json_file, out, save_img_name, save_label_name, save_viz_name, save_txt_name)
