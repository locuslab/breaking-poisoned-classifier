from PIL import Image
import random

import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import numpy as np
import time
import copy
import logging
import configparser
from dataset import LabeledDataset
from alexnet_fc7out import NormalizeByChannelMeanStd

import matplotlib
from matplotlib import pyplot as plt

import sys
sys.path.append("../../denoised-smoothing/code")
from architectures import get_architecture


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

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

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def adjust_learning_rate(optimizer, epoch):
    global lr
    """Sets the learning rate to the initial LR decayed 10 times every 10 epochs"""
    lr1 = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr1

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename)

def plot(img_datas, title, prediction=None, length=10, save_path=None):
    f, axarr = plt.subplots(3,3,figsize=(length,length))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.25)

    for i in range(3):
        for j in range(3):
            axarr[i,j].imshow(img_datas[i*3+j])
            if prediction is not None:
                axarr[i,j].set_title("%d"%prediction[i*3+j], fontsize=14)

            axarr[i,j].axis('off')

    f.suptitle(title, fontsize=16, y=0.94)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def get_patch(img, start_x, start_y, len_x, len_y):
    img = np.copy(img)
    return img[start_x:start_x+len_x:, start_y:start_y+len_y,:]

def predict(model, img_list):
    img_list_tmp = []
    for img in img_list:
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        
        img_list_tmp.append(img)

    inputs = torch.cuda.FloatTensor(np.concatenate(img_list_tmp, axis=0))
    
    with torch.no_grad():
        outputs = model(inputs)
        
    return outputs.argmax(1)

def test_model_under_patch(model, dataloaders, trigger, is_inception=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    patch_size = trigger.shape[-1]

    epoch_acc_sum = 0
    # Each epoch has a training and validation phase
    for i in range(10):
#         for phase in ['test', 'notpatched', 'patched']:
        for phase in ['patched']:
            model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.cuda()
                labels = labels.cuda()
                if phase == 'patched':
                    random.seed(i)
                    for z in range(inputs.size(0)):
                        start_x = random.randint(0, 224-patch_size-1)
                        start_y = random.randint(0, 224-patch_size-1)

                        inputs[z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger#

                with torch.no_grad():
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
        epoch_acc_sum += epoch_acc

    epoch_acc_avg = epoch_acc_sum / 10.
    print("Phase %s Accuracy %.4f"%(phase, epoch_acc_avg))

def plot_5x5(img_datas, title, prediction=None, length=10, save_path=None):
    f, axarr = plt.subplots(5,5,figsize=(length,length))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.25)

    for i in range(5):
        for j in range(5):
            axarr[i,j].imshow(img_datas[i][j])
            if prediction is not None:
                axarr[i,j].set_title("%d"%prediction[i*5+j], fontsize=14)

            axarr[i,j].axis('off')

    f.suptitle(title, fontsize=16, y=0.94)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def predict_5x5(model, img_list):
    img_list_tmp = []
    for img_l in img_list:
        for img in img_l:
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)
        
            img_list_tmp.append(img)

    inputs = torch.cuda.FloatTensor(np.concatenate(img_list_tmp, axis=0))
 
    with torch.no_grad():
        outputs = model(inputs)
        
    return outputs.argmax(1)

def load_config(cfg_id=1):
    EXPR_CFG_FILE = "cfg/experiment_%d.cfg"%cfg_id

    config = configparser.ConfigParser()
    config.read(EXPR_CFG_FILE)

    experimentID = config["experiment"]["ID"]

    options = config["visualize"]
    clean_data_root    = options["clean_data_root"]
    poison_root    = options["poison_root"]
    denoiser_path = options["denoiser_path"]
    arch        = options["arch"]
    eps         = int(options["eps"])
    noise_sd = float(options["noise_sd"])
    patch_size  = int(options["patch_size"])
    rand_loc    = options.getboolean("rand_loc")
    trigger_id  = int(options["trigger_id"])
    num_poison  = int(options["num_poison"])
    num_classes = int(options["num_classes"])
    batch_size  = int(options["batch_size"])

    options = config["poison_generation"]
    target_wnid = options["target_wnid"]
    source_wnid_list = options["source_wnid_list"].format(experimentID)
    num_source = int(options["num_source"])

    feature_extract = True

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = arch

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # logging.info(model_ft)

    # Transforms
    data_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            ])

    dataset_clean = LabeledDataset(clean_data_root + "/train",
                                   "data/{}/finetune_filelist.txt".format(experimentID), data_transforms)
    dataset_test = LabeledDataset(clean_data_root + "/val",
                                   "data/{}/test_filelist.txt".format(experimentID), data_transforms)
    dataset_patched = LabeledDataset(clean_data_root + "/val",
                                   "data/{}/patched_filelist.txt".format(experimentID), data_transforms)

    dataloaders_dict = {}
    dataloaders_dict['test'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                            shuffle=True, num_workers=4)
    dataloaders_dict['patched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
                                                            shuffle=False, num_workers=4)
    dataloaders_dict['notpatched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
                                                            shuffle=False, num_workers=4)

    trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),transforms.ToTensor(),])

    trigger = Image.open('data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
    trigger = trans_trigger(trigger).unsqueeze(0).cuda()

    checkpointDir = "finetuned_models/" + experimentID + "/" + str(arch) + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
                    "/patch_size_" + str(patch_size) + "/num_poison_" + str(num_poison) + "/trigger_" + str(trigger_id)

    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(normalize, model_ft)

    checkpoint = torch.load(os.path.join(checkpointDir, "poisoned_model.pt"), map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'])

    classifier = model
    classifier.eval()

    #######################################################################################
    # Load denoiser
    checkpoint = torch.load(os.path.join(denoiser_path, "checkpoint.pth.tar"))
    denoiser = get_architecture(checkpoint['arch'], 'imagenet')
    denoiser.load_state_dict(checkpoint['state_dict'])

    denoised_classifier = torch.nn.Sequential(denoiser.module, model)

    denoised_classifier = torch.nn.DataParallel(denoised_classifier).cuda()
    #######################################################################################

    denoised_classifier = denoised_classifier.cuda()

    return dataloaders_dict, classifier, denoised_classifier, trigger


def generate_smoothed_adv_binary(dataloader, attacker, model, num_noise_vec, noise_sd, step_size, count_max=9):
    clean_img_list = [] 
    adv_img_list = []

    count = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        count += inputs.shape[0]

        inputs, labels = inputs.cuda(), labels.cuda()

        shape = list(inputs.shape)
        shape[0] = inputs.shape[0] * num_noise_vec
        inputs = inputs.repeat((1, num_noise_vec, 1, 1)).view(shape)
        noise = torch.randn_like(inputs).cuda() * noise_sd

        labels = torch.zeros_like(labels).cuda()

        inputs_adv = attacker.attack(model, inputs, labels, noise=noise,
            num_noise_vectors=num_noise_vec,
            targeted=False,
            step_size=step_size)

        for i in range(int(inputs.shape[0]/num_noise_vec)):
            clean_img_list.append(np.transpose(inputs.data[i*num_noise_vec].cpu().numpy(), (1,2,0)))
            adv_img_list.append(np.transpose(inputs_adv.data[i*num_noise_vec].cpu().numpy(), (1,2,0)))

        if count >= count_max:
            break
    return clean_img_list, adv_img_list

def generate_smoothed_adv_multi_cls(dataloader, attacker, model, num_noise_vec, noise_sd, step_size):
    clean_img_list = [[],[],[],[],[]] 
    adv_img_list = [[],[],[],[],[]]

    counter = np.zeros(5)
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        counter[labels.item()] += 1
        if counter[labels.item()] >= 6:
            continue

        if np.all(counter >= 6):
            break

        ###############################################################
        shape = list(inputs.shape)
        shape[0] = inputs.shape[0] * num_noise_vec
        inputs = inputs.repeat((1, num_noise_vec, 1, 1)).view(shape)
        noise = torch.randn_like(inputs).cuda() * noise_sd
        ################################################################

        targets = labels
        inputs_adv = attacker.attack(model, inputs, targets, noise=noise,
            num_noise_vectors=num_noise_vec,
            targeted=False,
            step_size=step_size)

        for i in range(int(inputs.shape[0]/num_noise_vec)):
            clean_img_list[labels.item()].append(np.transpose(inputs.data[i*num_noise_vec].cpu().numpy(), (1,2,0)))
            adv_img_list[labels.item()].append(np.transpose(inputs_adv.data[i*num_noise_vec].cpu().numpy(), (1,2,0)))
    return clean_img_list, adv_img_list