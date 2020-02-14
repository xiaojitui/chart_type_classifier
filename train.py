#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pickle 
import os
import glob
import zipfile 

from scipy.misc import imresize
import cv2

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim


from prepare import *


def create_label(imgtypes):
    labels = {}

    i = 0
    for name in imgtypes:
        labels[name] = i
        i += 1
    return labels



def get_train_test(imgsamples, train_ratio = 0.95):

    idx = np.arange(len(imgsamples)) 
    np.random.shuffle(idx) 
    idx_1 = idx[:int(train_ratio*len(imgsamples))] 
    idx_2 = idx[int(train_ratio*len(imgsamples)):]
    img_train = [imgsamples[k] for k in idx_1]
    img_test = [imgsamples[k] for k in idx_2]
    
    return img_train, img_test



def prepare_data(data, path, labels, batch_size = 2, output_h = 224, output_w = 224, need_resize = True):
    
    X_data = []
    y_data = []
    
    idx = np.random.choice(len(data), batch_size)
    for i in idx:
        name = data[i].split('_')[0]
        label = labels[name]
        
        img = plt.imread(path + data[i])
        
        if need_resize:
            img_resize = resize_img(img, output_h = output_h, output_w = output_w)
        else:
            img_resize = img
    
     
        X_data.append(img_resize/127.5-1)
        y_data.append(label)
        
    X_data = np.stack(X_data, axis = 0)
    y_data = np.stack(y_data, axis = 0)
    return X_data, y_data



def construct_resnet():
    # Create graph
    X = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    y = tf.placeholder(tf.int32, shape=[None])
    training = tf.placeholder_with_default(False, shape=[])

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_50(X, num_classes=1001, is_training=False)

    resnet_saver = tf.train.Saver()
    prelogits = tf.squeeze(prelogits, axis=[1, 2])

    n_outputs = 10 # now we have 10 classes

    # connect logits to n_outputs
    with tf.name_scope("new_output_layer"):
        out1 = tf.layers.dense(prelogits, 256, activation = None, name="out1")
        ou1_drop = tf.layers.dropout(out1, 0.5, training=training)
        out2 = tf.layers.dense(ou1_drop, 64, activation = None, name="out2")
        out2_drop = tf.layers.dropout(out2, 0.5, training=training)
        logits = tf.layers.dense(out2_drop, n_outputs, name="new_logits")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")


    ## logits = tf.squeeze(logits, axis=[1, 2], name="new_logits")


    with tf.name_scope("train"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()

        # only train 'logits'
        #new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                     #scope="resnet_v2_50/logits|resnet_v2_50/block4|resnet_v2_50/block3|new_logits") 
        ### scope="resnet_v2_50/logits|out1|out2|new_logits"
        #training_op = optimizer.minimize(loss, var_list=new_vars)  
        training_op = optimizer.minimize(loss) 

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()



def construct_inception():
    X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="X")
    y = tf.placeholder(tf.int32, shape=[None])
    training = tf.placeholder_with_default(False, shape=[])
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=training)

    inception_saver = tf.train.Saver()

    prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])

    n_outputs = 10 

    # connect logits to n_outputs
    with tf.name_scope("new_output_layer"):
        new_logits = tf.layers.dense(prelogits, n_outputs, name="new_logits")
        Y_proba = tf.nn.softmax(new_logits, name="Y_proba")

    with tf.name_scope("train"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=new_logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()

        #new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="new_logits") ###### only train 'new_logits'
        #training_op = optimizer.minimize(loss, var_list=new_vars)                             ###### only train 'new_logits'
        training_op = optimizer.minimize(loss) 

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(new_logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver() 




def execute_resnet():
    with tf.Session() as sess:
        init.run()
        resnet_saver.restore(sess, './saved_weights/resnet_v2_50.ckpt')
        #saver.restore(sess, "./pre_trained/newmodel.ckpt")
        best_acc = 0
        for epoch in range(50):
            for iteration in range(50):
                X_batch, y_batch = prepare_data(img_train, path, labels, batch_size = 32)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True}) # need a training section 

            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})

            print(epoch, acc_train, acc_test)
            if acc_test > best_acc:
                best_acc = acc_test
                save_path = saver.save(sess, "./fine_tuned/newmodel.ckpt")

    saveweights = glob.glob('./fine_tuned/')
    with zipfile.ZipFile('./finetuned.zip', 'w') as myzip:
        for f in saveweights:  
            myzip.write(f)




def execute_inception():
    with tf.Session() as sess:
        init.run()
        inception_saver.restore(sess, './saved_weights/inception_v3.ckpt')
        #saver.restore(sess, "./pre_trained/newmodel.ckpt")
        best_acc = 0
        for epoch in range(50):
            for iteration in range(50):
                X_batch, y_batch = prepare_data(img_train, path, labels, batch_size = 32, output_h = 299, output_w = 299)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True}) # need a training section 

            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})

            print(epoch, acc_train, acc_test)
            if acc_test > best_acc:
                best_acc = acc_test
                save_path = saver.save(sess, "./fine_tuned/newmodel.ckpt")
    
    saveweights = glob.glob('./fine_tuned/')
    with zipfile.ZipFile('./finetuned.zip', 'w') as myzip:
        for f in saveweights:  
            myzip.write(f)





def train_resnet():
    inpath = './datasets'
    outpath = './processed_datasets'
    #save_img_to_zip(inpath, outpath) ## optional
    load_img(inpath, outpath)

    with open(os.path.join(outpath, 'img_can_use.pkl'), 'rb') as f:
        imgfiles = pickle.load(f)
        imgtypes = check_img_type(imgfiles)


    labels = create_label(imgtypes)   
    img_train, img_test = get_train_test(imgfiles, train_ratio = 0.95)
    X_train, y_train = prepare_data(img_train, path, labels, batch_size = 32)
    X_test, y_test = prepare_data(img_test, path, labels, batch_size = 40)
    
    tf.reset_default_graph()
    
    
    construct_resnet()
    execute_resnet()
    
    


def train_inception():
    inpath = './datasets'
    outpath = './processed_datasets'
    #save_img_to_zip(inpath, outpath) ## optional
    load_img(inpath, outpath)

    with open(os.path.join(outpath, 'img_can_use.pkl'), 'rb') as f:
        imgfiles = pickle.load(f)
        imgtypes = check_img_type(imgfiles)


    labels = create_label(imgtypes)   
    img_train, img_test = get_train_test(imgfiles, train_ratio = 0.95)
    X_train, y_train = prepare_data(img_train, path, labels, batch_size = 32, output_h = 299, output_w = 299)
    X_test, y_test = prepare_data(img_test, path, labels, batch_size = 40, output_h = 299, output_w = 299)
    
    tf.reset_default_graph()
    
    construct_inception()
    execute_inception()




if __name__ == '__main__':
    train_resnet()
	# train_inception()



