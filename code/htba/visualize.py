'''
This scripts finetunes a model on poisoned data and tests it on clean validation images and patched source images.
'''
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
import configparser
import glob
from tqdm import tqdm
from dataset import LabeledDataset
from alexnet_fc7out import NormalizeByChannelMeanStd
import skimage.io
from skimage import img_as_ubyte

from utils import set_parameter_requires_grad, initialize_model

import sys
sys.path.append("../../denoised-smoothing/code")
from architectures import get_architecture

from attacks import PGD_L2


config = configparser.ConfigParser()
config.read(sys.argv[1])

experimentID = config["experiment"]["ID"]

options = config["visualize"]
clean_data_root	= options["clean_data_root"]
poison_root	= options["poison_root"]
denoiser_path = options["denoiser_path"]
arch = options["arch"]
model_type = options["model_type"]
gpu         = int(options["gpu"])
eps = int(options["eps"])
patch_size  = int(options["patch_size"])
rand_loc    = options.getboolean("rand_loc")
trigger_id  = int(options["trigger_id"])
num_poison  = int(options["num_poison"])
num_classes = int(options["num_classes"])
batch_size  = int(options["batch_size"])
momentum 	= float(options["momentum"])
noise_sd = float(options["noise_sd"])
epsilon = float(options["epsilon"])
num_noise_vec = int(options["num_noise_vec"])
num_steps = int(options["num_steps"])
targeted = options.getboolean("targeted")
vis_dir = options["vis_dir"].format(experimentID, model_type, arch, epsilon, num_steps, num_noise_vec, noise_sd, targeted)

options = config["poison_generation"]
target_wnid = options["target_wnid"]
source_wnid_list = options["source_wnid_list"].format(experimentID)
num_source = int(options["num_source"])

checkpointDir = "finetuned_models/" + experimentID + "/" + arch + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
				"/patch_size_" + str(patch_size) + "/num_poison_" + str(num_poison) + "/trigger_" + str(trigger_id)

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
)

logging.info("Experiment ID: {}".format(experimentID))


# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = arch

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),transforms.ToTensor(),])

trigger = Image.open('data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
trigger = trans_trigger(trigger).unsqueeze(0).cuda()


# Test poisoned model
logging.info("Testing poisoned model...")
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Transforms
data_transforms = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		])

logging.info("Initializing Datasets and Dataloaders...")


# Poisoned dataset
saveDir = poison_root + "/" + experimentID + "/rand_loc_" +  str(rand_loc) + \
					"/patch_size_" + str(patch_size) + "/trigger_" + str(trigger_id)
filelist = sorted(glob.glob(saveDir + "/*"))

# sys.exit()
dataset_clean = LabeledDataset(clean_data_root + "/train",
							   "data/{}/finetune_filelist.txt".format(experimentID), data_transforms)
dataset_test = LabeledDataset(clean_data_root + "/val",
							  "data/{}/test_filelist.txt".format(experimentID), data_transforms)
dataset_patched = LabeledDataset(clean_data_root + "/val",
								 "data/{}/patched_filelist.txt".format(experimentID), data_transforms)
dataset_poison = LabeledDataset(saveDir,
								"data/{}/poison_filelist.txt".format(experimentID), data_transforms)

dataloaders_dict = {}

dataloaders_dict['test'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
														shuffle=True, num_workers=4)
dataloaders_dict['patched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
														   shuffle=False, num_workers=4)
dataloaders_dict['notpatched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
															  shuffle=False, num_workers=4)

logging.info("Number of clean images: {}".format(len(dataset_clean)))
logging.info("Number of poison images: {}".format(len(dataset_poison)))

normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = nn.Sequential(normalize, model_ft)

checkpoint = torch.load(os.path.join(checkpointDir, "%s_model.pt"%model_type), map_location='cuda:0')
model.load_state_dict(checkpoint['state_dict'])

#######################################################################################
# Load denoiser
checkpoint = torch.load(os.path.join(denoiser_path, "checkpoint.pth.tar"))
denoiser = get_architecture(checkpoint['arch'], 'imagenet')
denoiser.load_state_dict(checkpoint['state_dict'])

model = torch.nn.Sequential(denoiser.module, model)
#######################################################################################

model = model.cuda()

def visualize(model, dataloader, is_inception=False):
    attacker = PGD_L2(steps=num_steps, max_norm=epsilon)

    model.eval()   # Set model to evaluate mode

    running_corrects = 0

    count = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):

        inputs = inputs.cuda()
        labels = labels.cuda()

        ###############################################################
        # inputs = inputs.repeat((args.num_noise_vec, 1, 1, 1))
        shape = list(inputs.shape)
        shape[0] = inputs.shape[0] * num_noise_vec
        inputs = inputs.repeat((1, num_noise_vec, 1, 1)).view(shape)
        noise = torch.randn_like(inputs).cuda() * noise_sd

        targets = labels
        if not targeted:
            targets = torch.zeros_like(labels).cuda()
        ################################################################

        inputs_adv = attacker.attack(model, inputs, targets, noise=noise,
            num_noise_vectors=num_noise_vec,
            targeted=targeted,
            step_size=2 * epsilon / num_steps)

        for i in range(int(inputs_adv.shape[0]/num_noise_vec)):
            skimage.io.imsave(os.path.join(vis_dir, "%d_adv.jpg"%(count)), img_as_ubyte(np.transpose(inputs_adv[i*num_noise_vec].data.cpu().numpy(), (1,2,0))))
            skimage.io.imsave(os.path.join(vis_dir, "%d_clean.jpg"%(count)), img_as_ubyte(np.transpose(inputs[i*num_noise_vec].data.cpu().numpy(), (1,2,0))))
            count += 1

        if count >= 20:
            break

visualize(model, dataloaders_dict['notpatched'], is_inception=(model_name=="inception"))
