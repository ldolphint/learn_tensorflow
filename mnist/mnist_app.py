#encoding=utf-8

import tensorflow as tf
import mnist_forward
import mnist_backward
import numpy as np
from PIL import Image


def restore_model(testPicArr):
    #创建一个默认图，在该图中执行以下操作（多数操作和train中一样）
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        #实现一个滑动平均模型， 参数MOVING_AVERAGE_DECAY用于控制模型更新的速度，训练过程维护了一个影子变量
        variable_average = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            # 通过checkpoint文件定位到最新模型
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)


                preValue = sess.run(preValue, feed_dict = {x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28,28),Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L')) #灰度图
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255


    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready



def application():
    picSet = raw_input("input the addr_txt of pictures you want test:")
    picList = open(picSet, 'r').readlines()
    for testPic in picList:
        testPicArr = pre_pic(testPic.strip('\n'))
        preValue = restore_model(testPicArr)
        print "The prediction number is :", preValue



if __name__ == "__main__":
    application()

