'''
This scripts generates poisoned data using BadNet;
'''

import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import logging
import matplotlib.pyplot as plt
import cv2
import configparser

from PIL import Image
from alexnet_fc7out import alexnet, NormalizeByChannelMeanStd, resnet18
from dataset import PoisonGenerationDataset

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms

config = configparser.ConfigParser()
config.read(sys.argv[1])

experimentID = config["experiment"]["ID"]

options = config["poison_generation"]
data_root    = options["data_root"]
txt_root    = options["txt_root"]
arch        = options["arch"]
seed        = None
gpu         = int(options["gpu"])
epochs      = int(options["epochs"])
patch_size  = int(options["patch_size"])
rand_loc    = options.getboolean("rand_loc")
trigger_id  = int(options["trigger_id"])
num_iter    = int(options["num_iter"])
num_poison  = int(options["num_poison"])
logfile     = options["logfile"].format(experimentID, arch, rand_loc, patch_size, num_poison, trigger_id)
target_wnid = options["target_wnid"]
source_wnid_list = options["source_wnid_list"].format(experimentID)
num_source = int(options["num_source"])

saveDir_poison = "poison_data/" + experimentID + "/" + arch + "/rand_loc_" +  str(rand_loc) + \
                    '/patch_size_' + str(patch_size) + '/trigger_' + str(trigger_id)

if not os.path.exists(saveDir_poison):
    os.makedirs(saveDir_poison)

if not os.path.exists("data/{}".format(experimentID)):
    os.makedirs("data/{}".format(experimentID))

def main():
    #logging
    if not os.path.exists(os.path.dirname(logfile)):
            os.makedirs(os.path.dirname(logfile))

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(logfile, "w"),
        logging.StreamHandler()
    ])

    logging.info("Experiment ID: {}".format(experimentID))

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker()

def main_worker():
    global best_acc1

    if gpu is not None:
        logging.info("Use GPU: {} for training".format(gpu))

    for epoch in range(epochs):
        # run one epoch
        generate(epoch)

def save_image(img, fname):
    img = img.data.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img[: , :, ::-1]
    cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

def generate(epoch):
    since = time.time()

    # TRIGGER PARAMETERS
    trans_image = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      ])
    trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
                                        transforms.ToTensor(),
                                        ])

    trigger = Image.open('data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
    trigger = trans_trigger(trigger).unsqueeze(0).cuda(gpu)

    # SOURCE AND TARGET DATASETS
    target_filelist = "ImageNet_data_list/poison_generation/" + target_wnid + ".txt"

    # Use source wnid list
    if num_source==1:
        logging.info("Using single source for this experiment.")
    else:
        logging.info("Using multiple source for this experiment.")

    with open("data/{}/multi_source_filelist.txt".format(experimentID),"w") as f1:
        with open(source_wnid_list) as f2:
            source_wnids = f2.readlines()
            source_wnids = [s.strip() for s in source_wnids]

            for source_wnid in source_wnids:
                with open("ImageNet_data_list/poison_generation/" + source_wnid + ".txt", "r") as f2:
                    shutil.copyfileobj(f2, f1)

    source_filelist = "data/{}/multi_source_filelist.txt".format(experimentID)


    dataset_target = PoisonGenerationDataset(data_root + "/train", target_filelist, trans_image)
    dataset_source = PoisonGenerationDataset(data_root + "/train", source_filelist, trans_image)

    # SOURCE AND TARGET DATALOADERS
    train_loader_target = torch.utils.data.DataLoader(dataset_target,
                                                    batch_size=100,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    pin_memory=True)

    train_loader_source = torch.utils.data.DataLoader(dataset_source,
                                                      batch_size=100,
                                                      shuffle=True,
                                                      num_workers=8,
                                                      pin_memory=True)

    logging.info("Number of target images:{}".format(len(dataset_target)))
    logging.info("Number of source images:{}".format(len(dataset_source)))

    # USE ITERATORS ON DATALOADERS TO HAVE DISTINCT PAIRING EACH TIME
    iter_target = iter(train_loader_target)
    iter_source = iter(train_loader_source)

    num_poisoned = 0
    for i in range(len(train_loader_source)):

        # LOAD ONE BATCH OF SOURCE AND ONE BATCH OF TARGET
        (input1, path1) = next(iter_source)

        img_ctr = 0

        input1 = input1.cuda(gpu)

        for z in range(input1.size(0)):
            if not rand_loc:
                start_x = 224-patch_size-5
                start_y = 224-patch_size-5
            else:
                start_x = random.randint(0, 224-patch_size-1)
                start_y = random.randint(0, 224-patch_size-1)

            # PASTE TRIGGER ON SOURCE IMAGES
            input1[z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger

        for k in range(input1.size(0)):
            img_ctr = img_ctr+1

            fname = saveDir_poison + '/' + 'badnet_' + str(os.path.basename(path1[k])).split('.')[0] + '_' + 'epoch_' + str(epoch).zfill(2)\
                    + str(img_ctr).zfill(5)+'.png'

            save_image(input1[k].clone().cpu(), fname)
            num_poisoned +=1

if __name__ == '__main__':
    main()
