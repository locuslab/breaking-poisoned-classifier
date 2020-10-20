# Experiments
This directory contains the code for attacking poisoned classifiers as demonstrated in the paper. There are four directories: 
1. [badnet](badnet): code for training and attacking poisoned classifiers obtained by [BadNet](https://arxiv.org/abs/1708.06733)
2. [htba](htba): code for training and attacking poisoned classifiers obtained by [HTBA](https://arxiv.org/abs/1910.00033)
3. [clbd](clbd): code for training and attacking poisoned classifiers obtained by [CLBD](https://people.csail.mit.edu/madry/lab/cleanlabel.pdf)
4. [trojAI](trojAI): code for attacking poisoned classifiers in [trojAI dataset](https://pages.nist.gov/trojai/docs/data.html)

The code for generating poisoned data and training poisoned classifiers in directories [badnet](badnet), [htba](htba) and [cldb](clbd) is based on the open source repository [Hidden-Trigger-Backdoor-Attacks](https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks).