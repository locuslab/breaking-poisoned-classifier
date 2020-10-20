# Hidden Trigger Backdoor Attacks
## Overview
1. [cfg](cfg) contains the config files for poisoned classifiers.
2. [data](data) contains the dataset config and backdoor triggers.
3. The programs `generate_poisoned.py` and `finetune_and_test.py` are the codes for training poisoned classifier.
4. The notebook `breaking-poisoned-classifier.ipynb` contains the implementation of our attack methods.
5. The program `visualize.py` contains the code to generate adversarial examples of robustified smoothed classifiers. 

In [cfg](cfg), each config file corresponds to a poisoned classifier in the paper:
* `experiment_1.cfg`: HTBA poisoned binary classifier on ImageNet with Trigger A; 
* `experiment_2.cfg`: HTBA poisoned multi-class classifier on ImageNet with Trigger A;

In [data/triggers](data/triggers), `trigger_1.png` corresponds to Trigger A used in the paper.

## Getting Started
1. Prepare the dataset:
    ```
    python create_imagenet_filelist.py cfg/dataset.cfg
    ```
2. Generate poisoned data:
    ```
    python generate_poison.py cfg/experiment_1.cfg
    ```
   Train poisoned classifiers:
    ```
    python finetune_and_test.py cfg/experiment_1.cfg
    ```
   You can skip this step by downloading our trained poisoned classifiers (See step 3).
3. Download the trained poisoned classifiers from [here](https://drive.google.com/file/d/1kx8OaFVi7F4iW5N38I8GkbWeAGSPDmzU/view?usp=sharing). Then move the downloaded `htba_models.tar.gz` into this directory. Run `tar -xzvf htba_models.tar.gz` to extract the models. 
4. Start the jupyter notebook (requires CUDA): `jupyter notebook` and open `breaking-poisoned-classifier.ipynb`.  
5. Visualize adversarial examples of the *smoothed* classifier:
   ```
   python visualize.py cfg/experiment_1.cfg
   ```