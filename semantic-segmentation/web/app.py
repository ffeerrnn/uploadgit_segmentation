from flask import Flask,render_template,request
import numpy as np
import flask
# import tensorflow as tf
app = Flask(__name__)

@app.route('/',methods = ['POST','GET'])
def man():
    return render_template("home.html")

@app.route("/test",methods=["POST"])
def home():
    '''这个函数下面可以写你的预测程序，将最后的预测结果写在下面的A处即可'''
    return render_template("after.html", pred='识别结果为：%s' % (A))

if __name__ == '__main__':
    app.run()

