import argparse
import torch
import os 
import numpy as np 
import skimage.io 
import scipy.ndimage
import scipy.misc
from skimage import img_as_ubyte
import torch.backends.cudnn as cudnn

import sys
sys.path.append("../../denoised-smoothing/code")
from architectures import get_architecture

from attacks import PGD_L2

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--basepath', default="/data/datasets/trojai/trojai-round0-dataset/", type=str)
parser.add_argument('--epsilon', default=60, type=float)
parser.add_argument('--num-steps', default=10, type=int)
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors to use for finding adversarial examples. `m_train` in the paper.")
parser.add_argument('--no-grad-attack', action='store_true',
                    help="Choice of whether to use gradients during attack or do the cheap trick")
parser.add_argument('--denoiser', type=str, help="path to the denoiser")
parser.add_argument('--noise_sd', type=float, default=0.12)
parser.add_argument('--outdir', type=str)
parser.add_argument('--targeted', type=int, default=-1, 
                    help="-1 means the attack is untargeted, if it is 0-4 then it refers to the target in targeted attack.")
parser.add_argument('--step_size', type=float, default=0)
parser.add_argument('--id_start', type=int, default=0)
parser.add_argument('--id_end', type=int, default=1)
args = parser.parse_args()


def main(id_):
    BASE_PATH = args.basepath
 
    path_per_id = os.path.join(BASE_PATH, "id-%.8d"%id_)
    examples_dirpath = os.path.join(path_per_id, "example_data")
    modelpath = os.path.join(path_per_id, "model.pt")

    try:
        classifier = torch.load(modelpath)
    except:
        return
    model = classifier

    if args.denoiser is not None:
        checkpoint = torch.load(args.denoiser)
        denoiser = get_architecture(checkpoint['arch'], 'imagenet')
        denoiser.load_state_dict(checkpoint['state_dict'])

        model = torch.nn.Sequential(denoiser, classifier)

    model = model.cuda()
    model.eval()

    attacker = PGD_L2(steps=args.num_steps, max_norm=args.epsilon)

    count_per_class = np.zeros(5)

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith("png")]
    count = 0
    for fn in fns:
        count += 1

        path = os.path.split(fn)[-1]

        try:
            class_id = int(path[6])
        except:
            class_id = int(path[8])

        if count_per_class[class_id] >= 6:
            continue 

        if np.all(count_per_class >= 6):
            exit()

        # read the image (using skimage)
        img = skimage.io.imread(fn)

        if args.targeted == -1:
            save_dir = os.path.join(args.outdir, "id_%d"%id_, "pretrained_denoiser", "sigma_%f_noise_vec_%d_step_%d_eps_%d"%(args.noise_sd, args.num_noise_vec, args.num_steps, int(args.epsilon)), 
                                    "untargeted", "class_%d"%class_id)
        else:
            save_dir = os.path.join(args.outdir, "id_%d"%id_, "pretrained_denoiser", "sigma_%f_noise_vec_%d_step_%d_eps_%d"%(args.noise_sd, args.num_noise_vec, args.num_steps, int(args.epsilon)), 
                                    "targeted_%d"%args.targeted, "class_%d"%class_id)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        skimage.io.imsave(os.path.join(save_dir,"clean_%d.jpg"%(count_per_class[class_id])), img_as_ubyte(img))

        # convert to BGR (training codebase uses cv2 to load images which uses bgr format)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        img = np.stack((b, g, r), axis=2)

        img = np.transpose(img, (2, 0, 1))
        # convert to NCHW dimension ordering
        img = np.expand_dims(img, 0)
        # normalize the image
        img = img - np.min(img)
        img = img / np.max(img)
        # convert image to a gpu tensor
        batch_data = torch.FloatTensor(img)

        batch_data = batch_data.cuda()

        inputs = batch_data 
        ###################
        if args.targeted == -1:
            targets = torch.LongTensor([class_id]).cuda() 
            targeted = False 
        else:
            targets = torch.LongTensor([args.targeted]).cuda()
            targeted = True 
        ###################

        inputs = inputs.repeat((args.num_noise_vec, 1, 1, 1))
        noise = torch.randn_like(inputs, device="cuda") * args.noise_sd

        # Attack the smoothed classifier 
        inputs_adv = attacker.attack(model, inputs, targets, noise=noise,
            num_noise_vectors=args.num_noise_vec,
            targeted=targeted,
            no_grad=args.no_grad_attack,
            step_size = 2 * args.epsilon / args.num_steps)

        img_adv = np.transpose(inputs_adv[0].data.cpu().numpy(), (1,2,0))
        b = img_adv[:, :, 0]
        g = img_adv[:, :, 1]
        r = img_adv[:, :, 2]
        img_adv = np.stack((r, g, b), axis=2)

        skimage.io.imsave(os.path.join(save_dir, "adv_%d.jpg"%(count_per_class[class_id])), img_as_ubyte(img_adv))

        count_per_class[class_id] += 1

if __name__ == "__main__":
    for id_ in range(args.id_start, args.id_end):
        main(id_)