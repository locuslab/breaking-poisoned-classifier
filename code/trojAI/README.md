# TrojAI
## Overview
1. The notebook `breaking-poisoned-classifier.ipynb` contains a demonstration of our attack on the TrojAI dataset.
2. The program `visualize.py` contains the code for generating adversarial examples of robustified smoothed classifier.

## Getting Started
1. Download the TrojAI dataset following instructions from [here](https://pages.nist.gov/trojai/docs/data.html). In the paper, we evaluate our method using data from [Round 0](https://pages.nist.gov/trojai/docs/data.html#round-0-dry-run).
2. Start the jupyter notebook (requires CUDA): `jupyter notebook` and open `breaking-poisoned-classifier.ipynb`.  
3. Visualize adversarial examples of the *smoothed* classifier:
   ```
   CUDA_VISIBLE_DEVICES=0 python3 visualize.py \
    --denoiser ../../denoised-smoothing/denoisers/pretrained_imagenet_denoisers/dncnn_5epoch_lr1e-4/noise_1.00/checkpoint.pth.tar \
    --num-noise-vec 16 --num-steps 10 --noise_sd 1.00 --epsilon 20 --id_start 0 --id_end 100 \
    --outdir [DIR to SAVE RESULTS]
   ```
   where `id_start` and `id_end` specify the range of ids for classifiers whose adversarial examples are generated; `denoiser` specifies the path to the denoiser, the path given in the command is the default path if you download the denoisers in directory [denoised-smoothing(../denoised-smoothing) by following [instructions](https://github.com/microsoft/denoised-smoothing#getting-started).