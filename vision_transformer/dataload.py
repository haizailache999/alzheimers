import os
os.environ.setdefault['allreduce_post_accumulation']='True'
os.environ.setdefault['allreduce_post_accumulation_fp16']='True'
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import tensorflow as tf
#from tensorflow.keras import losses, optimizers, metrics, layers
#from tensorflow_addons.metrics import F1Score

import numpy as np
import pandas as pd
import time
import math
import json
import argparse
import psutil
from os import makedirs
from os.path import join
from DataGenerator import MRIDataGenerator, MRIDataGenerator_Simple
from dataAugmentation import MRIDataAugmentation

def train(args):
    READ_DIR='C:\\Users\\user\\uiuc_data\\data1\\'
    trainData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                     split='train',
                                     batchSize=args.batch_size * args.worst_sample,
                                     MCI_included=args.mci,
                                     MCI_included_as_soft_label=args.mci_balanced,
                                     idx_fold=args.idx_fold,
                                     augmented=args.augmented,
                                     augmented_fancy=args.augmented_fancy,
                                     dropBlock=args.dropBlock,
                                     dropBlockIterationStart=int(args.continueEpoch*1700/args.batch_size),
                                     gradientGuidedDropBlock=args.gradientGuidedDropBlock)


    validationData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                      batchSize=args.batch_size,
                                      idx_fold=args.idx_fold,
                                      split='val')

    testData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                batchSize=args.batch_size,
                                idx_fold=args.idx_fold,
                                split='test')
    for i in range(2):
        images, labels = trainData[i]
        print(images,labels)

def main(args):
    train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=50, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='Batch size during training per GPU')
    parser.add_argument('-a', '--action', type=int, default=0, help='action to take')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed')
    parser.add_argument('-i', '--idx_fold', type=int, default=0, help='which partition of data to use')
    parser.add_argument('-u', '--augmented', type=int, default=0, help='whether use augmentation or not')
    parser.add_argument('-g', '--augmented_fancy', type=int, default=0,
                        help='whether use the fancy, Alzheimer specific augmentation or not')
    parser.add_argument('-m', '--mci', type=int, default=0, help='whether use MCI data or not')
    parser.add_argument('-l', '--mci_balanced', type=int, default=0,
                        help='when using MCI, whether including it as a balanced data')
    parser.add_argument('-c', '--continueEpoch', type=int, default=0, help='continue from current epoch')
    parser.add_argument('-p', '--pgd', type=float, default=0, help='whether we use pgd (actually fast fgsm)')
    parser.add_argument('-n', '--minmax', type=int, default=0, help='whether we use min max pooling')
    parser.add_argument('-f', '--weights_folder', type=str, default='.', help='the folder weights are saved')
    parser.add_argument('-v', '--saliency_save_dir', type=str, default=READ_DIR + 'saliency_maps',
                        help='the folder to save saliency maps')
    parser.add_argument('-j', '--extract_features', type=int, default=0,
                        help='whether the model should exclude the last FC layer')
    parser.add_argument('-k', '--attack_visualization_save_dir', type=str, default=READ_DIR + 'attack_visualization',
                        help='the folder to save adversarial attack visualization')
    parser.add_argument('-w', '--activation_maximization_dir', type=str, default=READ_DIR + 'activation_maximizations',
                        help='the folder to save visualized activation maximizations')
    parser.add_argument('-z', '--visualize_feature_idx', type=int, default=0,
                        help='feature visualizing activation maximization, only 0 or 1 for if we consider the model prediction')
    parser.add_argument('-d', '--dropBlock', type=int, default=0,
                        help='whether we drop half of the information of the images')
    parser.add_argument('-o', '--gradientGuidedDropBlock', type=int, default=0,
                        help='whether we perform gradient guided dropBlock')
    parser.add_argument('-r', '--worst_sample', type=int, default=0, help='whether we use min max pooling')
    parser.add_argument('-y', '--consistency', type=float, default=0, help='whether we use min max pooling')
    parser.add_argument('-t', '--gpu', type=str, default=0,
                        help='specify maximum GPU ID we want to distribute the training to')

    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))

    if args.action == 0:
        main(args)