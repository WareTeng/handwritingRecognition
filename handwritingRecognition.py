#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Dengwei
@License :   (C) Copyright 2018-2019, SCUT
@Contact :   dw_hey@163.com
@Software:   PyCharm
@File    :   handwritingRecognition.py
@Time    :   2019/3/8 15:09
@Desc    :
"""

import tensorflow as tf
import sys
from PySide2.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PySide2.QtGui import QPainter, QPen, QFont
from PySide2.QtCore import Qt
from PIL import ImageGrab, Image
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def train():
    """
    采用LeNet卷积神经网络
    """
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x_, W_):
        return tf.nn.conv2d(x_, W_, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x_):
        return tf.nn.max_pool(x_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_drop = tf.placeholder(tf.float32, name='keep_drop')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_drop)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_conv')

    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 保存model
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_drop: 1.0})
            print("step {},training accuracy {}".format(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_drop: 0.5}, session=sess)
    # print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_drop: 1.0}))
    saver.save(sess, './saver/model.ckpt')


class drawingBoard(QWidget):
    def __init__(self):
        super(drawingBoard, self).__init__()

        # set height and width
        self.resize(284, 330)
        self.move(100, 100)
        self.setWindowTitle("drawing board")
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        # 按下鼠标跟踪鼠标事件
        self.setMouseTracking(False)
        # 用一个list来保持鼠标轨迹
        self.pos_xy = []

        # 绘制1px的画板边框
        self.label_draw = QLabel('', self)
        self.label_draw.setGeometry(2, 2, 280, 280)
        self.label_draw.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_draw.setAlignment(Qt.AlignCenter)

        self.label_result_name = QLabel('识别结果：', self)
        self.label_result_name.setGeometry(2, 290, 60, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(64, 290, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_recognize = QPushButton("识别", self)
        self.btn_recognize.setGeometry(110, 290, 50, 35)
        self.btn_recognize.clicked.connect(self.recognizePainter)

        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.setGeometry(170, 290, 50, 35)
        self.btn_clear.clicked.connect(self.clearPainter)

        self.btn_close = QPushButton("关闭", self)
        self.btn_close.setGeometry(230, 290, 50, 35)
        self.btn_close.clicked.connect(self.closePainter)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 18, Qt.SolidLine)  # 配置画笔，颜色、线宽、线型
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp
                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end

        painter.end()

    def mouseMoveEvent(self, event):
        pos_tmp = (event.pos().x(), event.pos().y())
        self.pos_xy.append(pos_tmp)
        self.update()

    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

    def recognizePainter(self):
        bbox = (104, 104, 380, 380)
        image = ImageGrab.grab(bbox)
        image = image.resize((28, 28), Image.ANTIALIAS)
        image.save('digit.png')
        recognize_result = self.recognize_image(image)
        self.label_result.setText(str(recognize_result))
        self.update()

    def clearPainter(self):
        self.pos_xy = []
        self.label_result.setText('')
        self.update()

    def closePainter(self):
        self.close()

    def recognize_image(self, image):
        image = image.convert('L')
        tv = list(image.getdata())
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]

        init = tf.global_variables_initializer()
        saver = tf.train.Saver

        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.import_meta_graph('./saver/model.ckpt.meta')
            saver.restore(sess, './saver/model.ckpt')

            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            keep_drob = graph.get_tensor_by_name("keep_drop:0")
            y_conv = graph.get_tensor_by_name("y_conv:0")

            prediction = tf.argmax(y_conv, 1)
            predint = prediction.eval(feed_dict={x: [tva], keep_drob: 1.0}, session=sess)
            print(predint[0])
        return predint[0]


if __name__ == '__main__':
    # train()
    app = QApplication(sys.argv)
    board = drawingBoard()
    board.show()
    app.exec_()
