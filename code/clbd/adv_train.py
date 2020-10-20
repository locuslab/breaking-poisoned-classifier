from PIL import Image
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import logging
import sys
import configparser
import glob
from tqdm import tqdm
from dataset import LabeledDataset
from alexnet_fc7out import NormalizeByChannelMeanStd

from utils import *

config = configparser.ConfigParser()
config.read(sys.argv[1])

experimentID = config["experiment"]["ID"]

options = config["adv_train"]
clean_data_root    = options["clean_data_root"]
arch        = options["arch"]
gpu         = int(options["gpu"])
epochs      = int(options["epochs"])
eps         = int(options["eps"])
alpha       = float(options["alpha"])
attack_iter = int(options["attack_iter"])
num_classes = int(options["num_classes"])
batch_size  = int(options["batch_size"])
lr            = float(options["lr"])
logfile     = options["logfile"].format(experimentID, arch, eps, alpha, attack_iter, lr, epochs)
momentum     = float(options["momentum"])

options = config["poison_generation"]
target_wnid = options["target_wnid"]
source_wnid_list = options["source_wnid_list"].format(experimentID)
num_source = int(options["num_source"])

checkpointDir = "adv_train/" + experimentID + "/" + str(arch) + "/eps_" + str(eps) + \
                "_alpha_" + str(alpha) + "_iter_" + str(attack_iter) + "_lr_" + str(lr) + "_epochs_" + str(epochs)

###############################
# Normalize the attack parameters
alpha = alpha / 255. 
eps = float(eps) / 255.
###############################

if not os.path.exists(os.path.dirname(checkpointDir)):
    os.makedirs(os.path.dirname(checkpointDir))

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


# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "alexnet"
# model_name = "resnet"
model_name = arch

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    test_acc_arr = np.zeros(num_epochs)
    adv_acc_arr = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch)
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test', 'adv']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = dataloaders["train"]
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = dataloaders["test"]

            running_loss = 0.0
            running_corrects = 0

            nn=1

            for ctr in range(0, nn):
                # Iterate over data.
                for inputs, labels in tqdm(dataloader):

                    inputs = inputs.cuda(gpu)
                    labels = labels.cuda(gpu)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train' or phase=="adv"):
                        if phase == "train" or phase == "adv":
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

                            delta = delta.detach()
                            outputs = model(inputs + delta)

                            loss = criterion(outputs, labels)
                            ##########################################################
                        elif phase == "test":
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset) / nn
            epoch_acc = running_corrects.double() / len(dataloader.dataset) / nn


            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'test':
                test_acc_arr[epoch] = epoch_acc
            elif phase == 'adv':
                adv_acc_arr[epoch] = epoch_acc

            # deep copy the model
            if phase == 'test':
                logging.info("Better model found!")
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Max Test Acc: {:4f}'.format(best_acc))
    logging.info('Last 10 Epochs Test Acc: Mean {:.3f} Std {:.3f} '
                 .format(test_acc_arr[-10:].mean(),test_acc_arr[-10:].std()))
    logging.info('Last 10 Epochs Adv Acc: Mean {:.3f} Std {:.3f} '
                 .format(adv_acc_arr[-10:].mean(),adv_acc_arr[-10:].std()))

    sort_idx = np.argsort(test_acc_arr)
    top10_idx = sort_idx[-10:]
    logging.info('10 Epochs with Best Acc- Test Acc: Mean {:.3f} Std {:.3f} '
                 .format(test_acc_arr[top10_idx].mean(),test_acc_arr[top10_idx].std()))
    logging.info('10 Epochs with Best Acc- Adv Acc: Mean {:.3f} Std {:.3f} '
                 .format(adv_acc_arr[top10_idx].mean(),adv_acc_arr[top10_idx].std()))


    # save meta into pickle
    meta_dict = {'Val_acc': test_acc_arr,
                 }

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, meta_dict


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        kwargs = {"transform_input": True}
        model_ft = models.inception_v3(pretrained=use_pretrained, **kwargs)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        logging.info("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def adjust_learning_rate(optimizer, epoch):
    global lr
    """Sets the learning rate to the initial LR decayed 10 times every 10 epochs"""
    lr1 = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr1

# Training dataset
# if not os.path.exists("data/{}/finetune_filelist.txt".format(experimentID)):
with open("data/{}/finetune_filelist.txt".format(experimentID), "w") as f1:
    with open(source_wnid_list) as f2:
        source_wnids = f2.readlines()
        source_wnids = [s.strip() for s in source_wnids]

    if num_classes==1000:
        wnid_mapping = {}
        all_wnids = sorted(glob.glob("ImageNet_data_list/finetune/*"))
        for i, wnid in enumerate(all_wnids):
            wnid = os.path.basename(wnid).split(".")[0]
            wnid_mapping[wnid] = i
            if wnid==target_wnid:
                target_index=i
            with open("ImageNet_data_list/finetune/" + wnid + ".txt", "r") as f2:
                lines = f2.readlines()
                for line in lines:
                    f1.write(line.strip() + " " + str(i) + "\n")

    else:
        for i, source_wnid in enumerate(source_wnids):
            with open("ImageNet_data_list/finetune/" + source_wnid + ".txt", "r") as f2:
                lines = f2.readlines()
                for line in lines:
                    f1.write(line.strip() + " " + str(i) + "\n")

        with open("ImageNet_data_list/finetune/" + target_wnid + ".txt", "r") as f2:
            lines = f2.readlines()
            for line in lines:
                f1.write(line.strip() + " " + str(num_source) + "\n")

# Test dataset
# if not os.path.exists("data/{}/test_filelist.txt".format(experimentID)):
with open("data/{}/test_filelist.txt".format(experimentID), "w") as f1:
    with open(source_wnid_list) as f2:
        source_wnids = f2.readlines()
        source_wnids = [s.strip() for s in source_wnids]


    if num_classes==1000:
        all_wnids = sorted(glob.glob("ImageNet_data_list/test/*"))
        for i, wnid in enumerate(all_wnids):
            wnid = os.path.basename(wnid).split(".")[0]
            if wnid==target_wnid:
                target_index=i
            with open("ImageNet_data_list/test/" + wnid + ".txt", "r") as f2:
                lines = f2.readlines()
                for line in lines:
                    f1.write(line.strip() + " " + str(i) + "\n")

    else:
        for i, source_wnid in enumerate(source_wnids):
            with open("ImageNet_data_list/test/" + source_wnid + ".txt", "r") as f2:
                lines = f2.readlines()
                for line in lines:
                    f1.write(line.strip() + " " + str(i) + "\n")

        with open("ImageNet_data_list/test/" + target_wnid + ".txt", "r") as f2:
            lines = f2.readlines()
            for line in lines:
                f1.write(line.strip() + " " + str(num_source) + "\n")

#####################################################################################################################
# Train clean model
logging.info("Training clean model...")
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
logging.info(model_ft)

# Transforms
data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        ])

logging.info("Initializing Datasets and Dataloaders...")


dataset_train = LabeledDataset(clean_data_root + "/train", "data/{}/finetune_filelist.txt".format(experimentID), data_transforms)
dataset_test = LabeledDataset(clean_data_root + "/val", "data/{}/test_filelist.txt".format(experimentID), data_transforms)

dataloaders_dict = {}
dataloaders_dict['train'] =  torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                         shuffle=True, num_workers=4)
dataloaders_dict['test'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                        shuffle=True, num_workers=4)
logging.info("Number of clean images: {}".format(len(dataset_train)))

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.

params_to_update = model_ft.parameters()
logging.info("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            logging.info(name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            logging.info(name)

optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum = momentum)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = nn.Sequential(normalize, model_ft)
model = model.cuda(gpu)

# Train and evaluate
model, meta_dict = train_model(model, dataloaders_dict, criterion, optimizer_ft,
                                  num_epochs=epochs, is_inception=(model_name=="inception"))

save_checkpoint({
                'arch': model_name,
                'state_dict': model.state_dict(),
                'meta_dict': meta_dict
                }, filename=os.path.join(checkpointDir, "adv_model.pt"))