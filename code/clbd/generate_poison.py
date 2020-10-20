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
import dill

from PIL import Image
from alexnet_fc7out import NormalizeByChannelMeanStd, alexnet, resnet18
from dataset import PoisonGenerationDataset

from utils import *

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
adv_model_path = options["adv_model_path"]
arch        = options["arch"]
seed        = None
gpu         = int(options["gpu"])
epochs      = int(options["epochs"])
patch_size  = int(options["patch_size"])
eps         = int(options["eps"])
lr          = float(options["lr"])
alpha       = float(options["alpha"])
rand_loc    = options.getboolean("rand_loc")
trigger_id  = int(options["trigger_id"])
attack_iter    = int(options["attack_iter"])
num_poison = int(options["num_poison"])
logfile     = options["logfile"].format(experimentID, arch, eps, alpha, attack_iter, rand_loc, patch_size, num_poison, trigger_id)
target_wnid = options["target_wnid"]
# target_idx = int(options["target_idx"])
source_wnid_list = options["source_wnid_list"].format(experimentID)
num_source = int(options["num_source"])
num_classes = int(options["num_classes"])

saveDir_poison = "poison_data/" + experimentID + "/" + arch + "/rand_loc_" +  str(rand_loc) + '/eps_' + str(eps) + \
                    '/attack_iter_' + str(attack_iter) + \
                    '/patch_size_' + str(patch_size) + '/trigger_' + str(trigger_id)
# saveDir_patched = "patched_data/" + experimentID + "/" + arch + "/rand_loc_" +  str(rand_loc) + '/eps_' + str(eps) + \
#                     '/patch_size_' + str(patch_size) + '/trigger_' + str(trigger_id)

alpha=alpha/255.
eps=eps/255.

if not os.path.exists(saveDir_poison):
    os.makedirs(saveDir_poison)
# if not os.path.exists(saveDir_patched):
#     os.makedirs(saveDir_patched)

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

    # create model
    # logging.info("=> using pre-trained model '{}'".format("alexnet"))
    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if arch == "alexnet":
        model = alexnet(pretrained=True)
    elif arch == "resnet":
        # model = resnet18(pretrained=True)
        model, input_size = initialize_model(arch, num_classes, False, use_pretrained=True)

    model.eval()
    model = nn.Sequential(normalize, model)

    checkpoint = torch.load(adv_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.cuda(gpu)

    for epoch in range(epochs):
        # run one epoch
        train(model, epoch)

# UTILITY FUNCTIONS
def show(img):
    npimg = img.numpy()
    # plt.figure()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

def save_image(img, fname):
    img = img.data.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img[: , :, ::-1]
    cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

def train(model, epoch):
    since = time.time()
    # AVERAGE METER
    losses = AverageMeter()

    # TRIGGER PARAMETERS
    trans_image = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      ])
    trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
                                        transforms.ToTensor(),
                                        ])

    # PERTURBATION PARAMETERS
    eps1 = (eps/255.0)
    lr1 = lr

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
                                                    batch_size=20,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    pin_memory=True)

    train_loader_source = torch.utils.data.DataLoader(dataset_source,
                                                      batch_size=20,
                                                      shuffle=True,
                                                      num_workers=8,
                                                      pin_memory=True)


    logging.info("Number of target images:{}".format(len(dataset_target)))
    logging.info("Number of source images:{}".format(len(dataset_source)))

    # USE ITERATORS ON DATALOADERS TO HAVE DISTINCT PAIRING EACH TIME
    iter_target = iter(train_loader_target)
    iter_source = iter(train_loader_source)

    num_poisoned = 0

    criterion = nn.CrossEntropyLoss()


    for i in range(len(train_loader_target)):

        # LOAD ONE BATCH OF SOURCE AND ONE BATCH OF TARGET
        (inputs, path) = next(iter_target)

        img_ctr = 0

        # input1 = input1.cuda(gpu)
        inputs = inputs.cuda(gpu)
        pert = nn.Parameter(torch.zeros_like(inputs, requires_grad=True).cuda(gpu))


        targets = torch.LongTensor([num_source]*inputs.shape[0]).cuda(gpu)
        # output, input_adv = model(input, targets, make_adv=True, **attack_kwargs)

        labels = targets
        ########################################################
        # Compute PGD adversarial examples
        delta = torch.zeros_like(inputs).cuda(gpu)
        delta.requires_grad = True 

        for _ in range(attack_iter):
            output = model(inputs+delta)
            loss = criterion(output, labels)

            loss.backward()

            grad = delta.grad.detach()
            delta.data = torch.clamp(delta + alpha * torch.sign(grad), min=-eps, max=eps)

            delta.data = clamp(delta, 0-inputs, 1-inputs)
            delta.grad.zero_()

        input_adv = inputs + delta
        ##########################################################

        # output_adv = model(input_adv, make_adv=False)

        for k in range(input_adv.size(0)):
            if not rand_loc:
                start_x = 224-patch_size-5
                start_y = 224-patch_size-5
            else:
                start_x = random.randint(0, 224-patch_size-1)
                start_y = random.randint(0, 224-patch_size-1)

            # PASTE TRIGGER ON SOURCE IMAGES
            input_adv[k, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger

            img_ctr = img_ctr+1
            input_pert = (input_adv[k].clone().cpu())

            fname = saveDir_poison + '/' + 'epoch_' + \
							str(epoch).zfill(2) + str(os.path.basename(path[k])).split('.')[0] + '_kk_' + str(img_ctr).zfill(5)+'.png'

            save_image(input_pert, fname)
            num_poisoned +=1

    print("number of poisoned images %d"%num_poisoned)
    time_elapsed = time.time() - since
    logging.info('Training complete one epoch in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(lr, iter):
    """Sets the learning rate to the initial LR decayed by 0.5 every 1000 iterations"""
    lr = lr * (0.5 ** (iter // 1000))
    return lr

if __name__ == '__main__':
    main()
