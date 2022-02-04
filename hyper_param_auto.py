#!/usr/bin/env python
# coding: utf-8

from ntpath import join
import sys
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
import tensorflow as tf
#import tf2onnx
#import onnxruntime as rt

import talos as tl

from download_dataset import *
from images_prepare import *
from masks_prepare import *
from model import *
from focal_loss import SparseCategoricalFocalLoss

import argparse

import json

sys.path.append(os.getcwd())

##### MAIN #####


class MyMeanIOU(MeanIoU):
    def __init__(self, num_classes, name=None, dtype=None):
        super(MyMeanIOU, self).__init__(
            num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_argmax = np.argmax(y_pred.numpy(), 3)
        y_pred_argmax = np.expand_dims(y_pred_argmax, axis=3)
        super(MyMeanIOU, self).update_state(y_true=y_true,
                                            y_pred=y_pred_argmax, sample_weight=sample_weight)


def write_json(save_path, history):
    with open(join(save_path, 'history.json'), 'w') as f:
        best_values = history.history
        if 'sparse_categorical_accuracy' in history.history.keys() and 'val_sparse_categorical_accuracy' in history.history.keys():
            best_values['best_sparse_categorical_accuracy'] = np.max(
                np.array(history.history['sparse_categorical_accuracy']))
            best_values['best_val_sparse_categorical_accuracy'] = np.max(
                np.array(history.history['val_sparse_categorical_accuracy']))
        if 'loss' in history.history.keys() and 'val_loss' in history.history.keys():
            best_values['best_loss'] = np.min(
                np.array(history.history['loss']))
            best_values['best_val_loss'] = np.min(
                np.array(history.history['val_loss']))
        if 'my_mean_iou' in history.history.keys() and 'val_my_mean_iou' in history.history.keys():
            best_values['best_mean_iou'] = np.max(
                np.array(history.history['my_mean_iou']))
            best_values['best_val_mean_iou'] = np.max(
                np.array(history.history['val_my_mean_iou']))

        json.dump(best_values, f, indent=2)


def write_config(save_path, args):
    with open(join(save_path, 'config.json'), 'w') as f:
        json.dump(args, f, indent=2)


def build_model(x_train, y_train, x_val, y_val, params):
    data_size = (512, 512)
    # Initialize model
    model = UNET(input_shape=(data_size[1], data_size[0], 3),
                 last_activation='softmax', num_classes=params['classes'])

    selected_losses = params['loss']

    # Optimizer
    optimizer = Adam(learning_rate=params['lr'])

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'categorical_accuracy', MeanIoU(num_classes=2), dice_coef])       # binary segmentation
    # metrics=['accuracy', 'sparse_categorical_accuracy', MeanIoU(num_classes=num_cls), dice_coef]
    model.compile(optimizer=optimizer, loss=selected_losses, metrics=[
                  'sparse_categorical_accuracy', MyMeanIOU(num_classes=params['classes'])], run_eagerly=True)

    callbackES = tl.utils.early_stopper(epochs=params['patience'])
    callbackSave = ModelCheckpoint(filepath=join(
        params['model_path'], "best_model.h5"), save_best_only=True)

    print("Training model ...")

    history = model.fit(x_train, y_train,
                        batch_size=params['batch'],
                        epochs=params['epochs'],       # num_epc
                        callbacks=[callbackES, callbackSave],
                        verbose=2,
                        validation_data=(x_val, y_val))

    return history, model


def main(args):
    # Input
    dataset_path = args.dataset_path
    img_path = os.path.join(dataset_path, "images/")
    msk_path = os.path.join(dataset_path, "masks/")
    img_tst_path = os.path.join(dataset_path, "test_images/")
    mask_tst_path = os.path.join(dataset_path, "test_masks/")
    # Output
    save_path = args.save_path  # "DataNemo/results/inference"
    if args.model_path == None:
        model_path = save_path
    else:
        model_path = args.model_path  # "checkpoints/uNet_171rgb_5cls_front"

    # Params
    data_size = (512, 512)
    # [6 teeth + 1 background] - [2 pairs + 2 cent incisors + 1 background]
    num_cls = args.num_classes
    num_epc = args.epochs
    # save final or checkpoints in TF format
    save_tf = model_path != None
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

    # TRAINING
    if args.patience == None:
        args.patience = int(num_epc / 10)

        # Only applied if patience is calculated
        if args.patience < 5:
            args.patience = 5

    # Training experiments
    p = {
        'lr': [0.1, 0.01, 0.001],
        'model_path': [args.save_path],
        'classes': [2],
        'batch': [args.batch],
        'epochs': [args.epochs],
        'patience': [args.patience],
        'loss': [sparse_categorical_crossentropy, SparseCategoricalFocalLoss(gamma=1), SparseCategoricalFocalLoss(gamma=2), SparseCategoricalFocalLoss(gamma=4)]
    }

    experiment = tl.Scan(x=img_trn,
                y=mask_trn,
                x_val=img_tst,
                y_val=mask_tst,
                model=build_model,
                params=p,
                experiment_name=args.experiment,
                reduction_metric='val_my_mean_iou')

    tl.Analyze(experiment)

    tl.Evaluate(experiment)


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


def plot_graphics(history, model_name_path):
    # sparse_categorical_accuracy
    if 'sparse_categorical_accuracy' in history.history.keys() and 'val_sparse_categorical_accuracy' in history.history.keys():
        plt.plot(history.history['sparse_categorical_accuracy'])
        plt.plot(history.history['val_sparse_categorical_accuracy'])
        plt.title('model sparse_categorical_accuracy')
        plt.ylabel('sparse_categorical_accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(
            join(model_name_path, 'sparse_categorical_accuracy_log.png'))
        plt.clf()

    # History loss
    if 'loss' in history.history.keys() and 'val_loss' in history.history.keys():
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(join(model_name_path, 'loss_log.png'))

        plt.clf()

    # summarize history for loss
    if 'my_mean_iou' in history.history.keys() and 'val_my_mean_iou' in history.history.keys():
        plt.plot(history.history['my_mean_iou'])
        plt.plot(history.history['val_my_mean_iou'])
        plt.title('model mean_iou')
        plt.ylabel('mean_iou')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(join(model_name_path,  'mean_iou_log.png'))


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
    parser.add_argument(
        '--patience', help='Learning rate used in the optimizer', type=int, default=5)
    parser.add_argument(
        '--experiment', help='Experiment name used', type=str, default='test')
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    main(args)
