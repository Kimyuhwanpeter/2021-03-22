# -*- coding:utf-8 -*-
from random import random, shuffle
from collections import Counter
from age_model_V3 import *

import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 256,
                           
                           "load_size": 276,
                           
                           "batch_size": 128,

                           "num_classes": 86,
                           
                           "epochs": 500,
                           
                           "lr": 0.02,
                           
                           "tr_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/train.txt",
                           
                           "tr_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI/",
                           
                           "te_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/test.txt",
                           
                           "te_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI/",
                           
                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpaoint": "",
                           
                           "graphs": ""})

optim = tf.keras.optimizers.SGD(FLAGS.lr)

def tr_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [128 + 20, 88 + 20])
    img = tf.image.random_crop(img, [128, 88, 3])

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
    img = tf.image.per_image_standardization(img)

    lab = lab_list - 2
    lab = tf.one_hot(lab, FLAGS.num_classes)

    return img, lab

def te_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [128, 88])

    img = tf.image.per_image_standardization(img)

    lab = lab_list - 2

    return img, lab

@tf.function
def run(model, images, training=True):
    return model(images, training=training)


def cal_loss(images, labels, model):

    with tf.GradientTape() as tape:

        logits = run(model, images, True)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def test_mae(images, labels, model):

    logits = run(model, images, False)
    prob = tf.nn.softmax(logits, -1)
    predict = tf.cast(tf.argmax(prob, 1), tf.int32)

    ae = tf.reduce_sum(tf.abs(predict - labels))

    return ae

def main():
    model = modified_GEINet(input_shape=(128, 88, 3),
                     num_classes=FLAGS.num_classes)

    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("* Restored the latest checkpoint!!!!! *")

    if FLAGS.train:
        count = 0

        tr_img = np.loadtxt("D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_train.txt", dtype="<U300", skiprows=0, usecols=0)
        tr_img = [FLAGS.tr_img_path + img + ".png"for img in tr_img]
        tr_lab = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)
        
        te_img = np.loadtxt("D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_test_fix.txt", dtype="<U300", skiprows=0, usecols=0)
        te_img = [FLAGS.te_img_path + img + ".png" for img in te_img]
        te_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.map(te_func)
        te_gener = te_gener.batch(137)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):

            T = list(zip(tr_img, tr_lab))
            shuffle(T)
            tr_img, tr_lab = zip(*T)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.map(tr_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(tr_img) // FLAGS.batch_size

            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)
                
                loss = cal_loss(batch_images, batch_labels, model)
                
                if count % 10 ==0:
                    print("【 Total loss = {}, step: {} [{}/{}], epoch: {} 】".format(loss, count + 1, step + 1, tr_idx, epoch))

                if count % 100 == 0 and count != 0:
                    te_iter = iter(te_gener)
                    ae = 0
                    for i in range(len(te_img) // 137):
                        image, label = next(te_iter)

                        ae += test_mae(image, label, model)

                        if i % 10 == 0:
                            print("* ⇒⇒⇒ part MAE = {} *".format(ae / ((i + 1)*137) ))

                    print("* {} step(s) ⇒⇒⇒ total MAE = {} *".format(count, ae / len(te_img)))


                count += 1

if __name__ == "__main__":
    main()