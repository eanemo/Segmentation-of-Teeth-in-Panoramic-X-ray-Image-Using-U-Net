#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.metrics import MeanIoU, BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import tensorflow as tf
#import tf2onnx
#import onnxruntime as rt

from download_dataset import *
from images_prepare import *
from masks_prepare import *
from model import *

import argparse

sys.path.append(os.getcwd())

##### MAIN #####


def main(args):
    # Input
    dataset_path = args.dataset_path
    img_path = os.path.join(dataset_path, "images/")
    msk_path = os.path.join(dataset_path, "masks/")
    img_tst_path = os.path.join(dataset_path, "test_images/")
    mask_tst_path = os.path.join(dataset_path, "test_masks/")
    # Output
    save_path = args.save_path  # "DataNemo/results/inference"
    model_path = args.model_path  # "checkpoints/uNet_171rgb_5cls_front"
    save_model_path = model_path + datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
    # Params
    data_size = (512, 512)
    # [6 teeth + 1 background] - [2 pairs + 2 cent incisors + 1 background]
    num_cls = args.num_classes
    num_epc = args.epochs
    save_tf = model_path != None                 # save final or checkpoints in TF format
    save_onnx = False               # save final or checkpoints in onnx format
    save_checkpoints = False        # save model for every 10 epochs
    save_prediction = True      # save test inference masks

    # Prepare output directories
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Prepare Image X and annotation Y data
    img_trn, img_trn_sizes, img_trn_dirs = pre_rgb_images_nemo(
        data_size, img_path)
    print("[Reading] Images training data shape:", img_trn.shape)
    img_tst, img_tst_sizes, img_tst_dirs = pre_rgb_images_nemo(
        data_size, img_tst_path)
    print("[Reading] Images test data shape:", img_tst.shape)

    # Lectura y MASK a one hot encoding donde cada imagen [w, h, c] donde c es número de clases (1 plano para cada clase)
    # mask_trn = prepare_masks_onehot(msk_path, num_cls, data_size, remove_background=False)
    # mask_trn = prepare_masks_nemo(data_size, msk_path)
    mask_trn = prepare_masks_sparse(msk_path, data_size)
    print("[Reading] Mask training data shape:", mask_trn.shape)
    # mask_tst = prepare_masks_onehot(mask_tst_path, num_cls, data_size, remove_background=False)
    # mask_tst = prepare_masks_nemo(data_size, mask_tst_path)
    mask_tst = prepare_masks_sparse(mask_tst_path, data_size)
    print("[Reading] Mask test data shape:", mask_tst.shape)

    # Normalize data [0, 1] -> Normalize images, MASKS sometimes
    img_trn = np.float32(img_trn/255)
    img_tst = np.float32(img_tst/255)
    # mask_trn = np.float32(mask_trn/255)
    # mask_tst = np.float32(mask_tst/255)

    # Initialize model
    model = UNET(input_shape=(data_size[1], data_size[0], 3),
                 last_activation='softmax', num_classes=num_cls)
    # model.summary()

    # TRAINING
    callbackES = EarlyStopping(monitor='loss',  patience=10)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'categorical_accuracy', MeanIoU(num_classes=2), dice_coef])       # binary segmentation
    # metrics=['accuracy', 'sparse_categorical_accuracy', MeanIoU(num_classes=num_cls), dice_coef]
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[
                  'sparse_categorical_accuracy'])

    # Training by blocks
    num_blocks = int(num_epc/10)
    for nblock in range(num_blocks):
        print("****** TRAINING BLOCK: {}-{}/{}".format((nblock)
              * 10, (nblock+1)*10, num_epc))
        history = model.fit(img_trn, mask_trn,
                            batch_size=args.batch,
                            epochs=10,       # num_epc
                            callbacks=[callbackES],
                            verbose=2,
                            validation_data=(img_tst, mask_tst))

        stime = time.time()
        predict_img = model.predict(img_tst)
        elapsed_time = time.time() - stime
        print("Inference done:", predict_img.shape,
              '- Inference time:', elapsed_time, )

        # Evaluation
        predict_img_argmax = np.argmax(predict_img, 3)
        predict_img_argmax = np.expand_dims(predict_img_argmax, axis=3)

        iou = MeanIoU(num_classes=num_cls)
        iou.update_state(mask_tst, predict_img_argmax)
        print("Evaluation MeanIoU:", iou.result().numpy())
        iou_values = np.array(iou.get_weights()).reshape(num_cls, num_cls)
        print(get_class_iou(iou_values, num_cls))

        # SAVE MODEL
        if save_checkpoints or (nblock+1) == num_blocks:
            save_model_path_block = save_model_path + \
                "_{}epc".format((nblock+1)*10)
            # SAVE whole model in SavedModel tf format
            if save_tf:
                model.save(save_model_path_block + '.h5')  # Save in h5 format
                # Save in SavedModel tf format
                model.save(save_model_path_block, save_format='tf')
                print("*** SAVE *** Model TF format saved in checkpoints/ directory:",
                      save_model_path_block)

            # Convert to ONNX
            if save_onnx:
                spec = (tf.TensorSpec(
                    (None, data_size[1], data_size[0], 3), tf.float32, name="input"),)
                output_path = save_model_path_block + ".onnx"
                model_proto, _ = tf2onnx.convert.from_keras(
                    model, input_signature=spec, opset=13, output_path=output_path)
                print("*** SAVE *** Model ONNX format saved in checkpoints/ directory:",
                      save_model_path_block)

    # Plot metric after training fit function - Problem with for loop
    #plot_dice_training(history, save_model_path)

    # Inference
    if save_prediction:
        print("****** Predictions saved in directory:", save_path)
        # Save masks per class
        for i in range(predict_img.shape[0]):
            img_name = os.path.splitext(img_tst_dirs[i])[0]
            masks = np.array(predict_img[i, :, :, :]*255)
            for p in range(masks.shape[2]):
                mskpath = os.path.join(
                    save_path, '{}_mask_{}.png'.format(img_name, p))
                cv2.imwrite(mskpath, masks[:, :, p])
                print("Mask plane: {}".format(mskpath))

        # Agregate masks in the final mask
        predict_img = np.argmax(predict_img, 3)
        # print("Mask shape post argmax:", predict_img.shape)

        # Save aggregated mask
        for i, img in enumerate(img_tst_dirs):
            img_name = os.path.splitext(img)[0]
            mskpath = os.path.join(save_path, 'mask_{}.png'.format(img_name))
            mask = predict_img[i, :, :]
            cv2.imwrite(mskpath, mask)
            print("Mask: {} --unique--> {}".format(mskpath, np.unique(mask)))


##### Dice Coefficient implementation #####
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


##### Plot dice coef. and loss on images from model history #####
def plot_dice_training(history, model_name_path):
    # summarize history for accuracy
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice coeficint')
    plt.ylabel('dice coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_name_path + '-dice_log.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_name_path + '-loss_log.png')

##### Plot IoU and loss on images from model history #####


def plot_iou_training(history, model_name_path):
    # summarize history for accuracy
    plt.plot(history.history['mean_io_u'])
    plt.plot(history.history['val_mean_io_u'])
    plt.title('model dice IoU')
    plt.ylabel('IoU')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_name_path + '-iou_log.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_name_path + '-loss_log.png')

# Get IoU per class from confusion matrix


def get_class_iou(conf_matrix, num_classes):
    iou = np.zeros(num_classes)
    for i in range(num_classes):
        wrong_pool = 0
        for j in range(num_classes):
            if i != j:
                wrong_pool += conf_matrix[i, j]
                wrong_pool += conf_matrix[j, i]
        iou[i] = conf_matrix[i, i] / (conf_matrix[i, i]+wrong_pool)
    return iou


#################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Entrenamiento de la red UNet.')
    parser.add_argument('dataset_path', help='Source dir del dataset')
    parser.add_argument(
        'save_path', help='Dirección de salida para guardar el resultado')
    parser.add_argument(
        '--model_path', help='Dirección dónde se almacenará el modelo', required=False)
    parser.add_argument('--verbose', dest='verbose',
                        action='store_true', help="Mostrar informacion del proceso")
    parser.add_argument('--batch', type=int,
                        help="Tamaño del batch", default=8)
    parser.add_argument('--num_classes', type=int,
                        help="Número de clases de la segmentación (classes + fondo)", default=2)
    parser.add_argument('--epochs', type=int,
                        help="Número de épocas del entrenamiento", default=100)

    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    main(args)
