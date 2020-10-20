import os
import torch
import skimage.io
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

import sys
sys.path.append("../../denoised-smoothing/code")
from architectures import get_architecture

def load_model(clf_id=0):
    denoiser_path = os.environ["denoiser_path"]
    clf_path = os.path.join(os.environ["trojai"], "id-%.8d"%clf_id, "model.pt")

    checkpoint = torch.load(denoiser_path)
    denoiser = get_architecture(checkpoint['arch'], 
                                'imagenet')
    denoiser.load_state_dict(checkpoint['state_dict'])
    ##################################################################
    # Load classification model
    classifier = torch.load(clf_path)
    denoised_classifier = torch.nn.Sequential(denoiser.module, classifier)

    denoised_classifier = torch.nn.DataParallel(denoised_classifier).cuda()
    denoised_classifier.eval()

    return classifier, denoised_classifier

def load_data(clf_id=0):
    base = os.path.join(os.environ["trojai"], "id-%.8d"%clf_id, "example_data")

    img_list = []
    label_list = []

    for class_id in range(5):
        for i in range(5):
            imgpath = os.path.join(base, "class_%d_example_%d.png"%(class_id, i))
            img = skimage.io.imread(imgpath)
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            img = np.stack((b, g, r), axis=2)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)
            img = img - np.min(img)
            img = img / np.max(img)

            img_list.append(img)
            label_list.append(class_id)

    inputs = torch.cuda.FloatTensor(np.concatenate(img_list, axis=0))
    labels = torch.cuda.LongTensor(np.array(label_list))

    return inputs, labels

def get_minibatches(batch, num_batches):
    X = batch[0]
    y = batch[1]

    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]

    if len(X) > (i+1)*batch_size:
        yield X[(i+1)*batch_size : len(X)], y[(i+1)*batch_size : len(y)]

def bgr2rgb(img):
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    img = np.stack((r, g, b), axis=2)
    return img

def generate_adv(model, inputs, labels, attacker, num_noise_vec, noise_sd, step_size):
    mini_batches = get_minibatches([inputs, labels], 8)

    clean_img_list = []
    adv_img_list = []
    for inputs, targets in mini_batches:
        inputs = inputs.cuda()
        targets = targets.cuda()

        # inputs = inputs.repeat((args.num_noise_vec, 1, 1, 1))
        shape = list(inputs.shape)
        shape[0] = inputs.shape[0] * num_noise_vec
        inputs = inputs.repeat((1, num_noise_vec, 1, 1)).view(shape)
        noise = torch.randn_like(inputs).cuda() * noise_sd

        # Attack the smoothed classifier 
        inputs_adv = attacker.attack(model, inputs, targets, noise=noise,
            num_noise_vectors=num_noise_vec,
            targeted=False,
            step_size=step_size)

        for i in range(int(inputs.shape[0]/num_noise_vec)):
            clean_img = np.transpose(inputs.data[i*num_noise_vec].cpu().numpy(), (1,2,0))
            adv_img = np.transpose(inputs_adv.data[i*num_noise_vec].cpu().numpy(), (1,2,0))
            clean_img_list.append(bgr2rgb(clean_img))
            adv_img_list.append(bgr2rgb(adv_img))

    return clean_img_list, adv_img_list

def test_model_under_patch(model, patch, clf_id, target_class, start_x=0, start_y=0):
    patch_size = patch.shape[-1]
    model.eval()   # Set model to evaluate mode

    # change to bgr format
    patch = patch[:, [2,1,0], :, :]

    running_corrects = 0

    datapath = os.path.join(os.environ["trojai"], "id-%.8d"%clf_id, "example_data")
    count = 0
    # Iterate over data.
    for filepath in os.listdir(datapath):
        if not filepath.endswith(".png"):
            continue

        class_id = int(filepath[6])

        if target_class == class_id:
            continue

        count += 1

        img = skimage.io.imread(os.path.join(datapath, filepath))
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        img = np.stack((b, g, r), axis=2)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img - np.min(img)
        img = img / np.max(img)

        inputs = torch.FloatTensor(img).cuda()

        for z in range(inputs.shape[0]):
            inputs[z, :, start_x:start_x+patch_size, start_y:start_y+patch_size] = patch

        with torch.no_grad():
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            running_corrects += int(preds.item() == target_class)

    epoch_acc = float(running_corrects) / float(count)
    print("Accuracy under current trigger: %.4f"%(epoch_acc))

    return inputs

def get_patch(img, start_x, start_y, len_x, len_y):
    img = np.copy(img)
    return img[start_x:start_x+len_x:, start_y:start_y+len_y,:]

def plot_5x5(img_datas, title, prediction=None, length=10, save_path=None):
    f, axarr = plt.subplots(5,5,figsize=(length,length))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.25)

    for i in range(5):
        for j in range(5):
            axarr[i,j].imshow(img_datas[i*5+j])
            if prediction is not None:
                axarr[i,j].set_title("%d"%prediction[i*5+j], fontsize=14)

            axarr[i,j].axis('off')

    f.suptitle(title, fontsize=16, y=0.94)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def predict_5x5(model, img_list):
    img_list_tmp = []
    for img in img_list:
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        img = np.stack((b, g, r), axis=2)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)

        img_list_tmp.append(img)

    inputs = torch.cuda.FloatTensor(np.concatenate(img_list_tmp, axis=0))

    with torch.no_grad():
        outputs = model(inputs)

    return outputs.argmax(1)