import argparse
import json
import os
import os.path as osp
import warnings


import yaml

import base64

import io
import PIL.Image
import numpy as np

from labelme_utils.image import img_b64_to_arr
from labelme_utils.shape import shapes_to_label
from labelme_utils.shape import

def main():
    json_file = "/home/liufang/project/xbq20.json"

    out_dir = "/home/liufang/project/"

    data = json.load(open(json_file))


    imageData = data['imageData']
    imageData = base64.b64encode(imageData).decode('utf-8')
    img = img_b64_to_arr(imageData)

    label_name_to_value = {'_background_': 0}
    for shape in data['shapes']:
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    # label_values must be dense
    label_values, label_names = [], []
    for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
        label_values.append(lv)
        label_names.append(ln)
    assert label_values == list(range(len(label_values)))

    lbl = shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    captions = ['{}: {}'.format(lv, ln)
                for ln, lv in label_name_to_value.items()]
    lbl_viz = draw_label(lbl, img, captions)

    out_dir = osp.basename("0").replace('.', '_')
    out_dir = osp.join(osp.dirname("0"), out_dir)
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    # cv2.imwrite(osp.join(out_dir, 'img.png'), lbl)
    PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
    utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
    # cv2.imwrite(osp.join(out_dir, 'label_viz.png'), lbl_viz)

    with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')

    warnings.warn('info.yaml is being replaced by label_names.txt')
    info = dict(label_names=label_names)
    with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
        yaml.safe_dump(info, f, default_flow_style=False)

    print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()
