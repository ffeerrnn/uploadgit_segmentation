import os
import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from fun import cfg, inference

app = Flask(__name__)
CORS(app)  # 解决跨域问题


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    # image = request.files["file"]
    # img_bytes = image.read()
    # info = get_prediction(image_bytes=img_bytes)

    # image = request.files["file"]
    # print("image", image.filename)
    # prediction, pix_num = inference(cfg, image.filename)
    # print("prediction", prediction.shape)
    print('done!')
    text = "class:1 probability:1"
    info = {"result": text}
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
