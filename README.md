# SACB: Tackling Label Noise in a Self-Adaptive Class-Balanced Manner
**Abstract:** Label noise is inevitable in the real-world dataset, which makes deep neural networks overfit seriously and hurts the model performance dramatically.
The excellent performance of current methods for learning with label noise is based on prior solid knowledge, such as noise rate, artificially preset threshold, and a well-labeled subset. However, this prior solid knowledge is difficult to estimate and obtain in real-world scenarios. Specifically, we design an innovative elf-adaptive Class-balanced sample Selection (SCS) method. Dynamically updated global thresholds and class-based local thresholds of SCS are estimated to divide the clean and noisy subsets based on given labels’ prediction probability and statistical values. Besides, we propose a Self-adaptive Class-balanced sample Re-weighting (SCR) method. A dynamic truncated normal distribution is estimated to assign different weights to different corrected samples based on their confidence to alleviate the deviation of correcting label class distribution. Finally, we employ Consistency Regularization (CR) between the sample’s strong data augmentation predictions and the sample’s weak data augmentation correction labels to mitigate the influence of a few inevitable noisy samples in the clean subset. Extensive experiment results on synthetic and real-world datasets demonstrate our proposed method's effectiveness and superiority, especially when the labeled data are extremely noisy (e.g., Symmetric-80%).

# Pipeline

![framework](Figure2.png)

# Installation
```
pip install -r requirements.txt
```

# Datasets
Currently three datasets are supported: CIFAR10, CIFAR100 and Clothing1M
Synthetic datasets are mainly derived from CIFAR10 and CIFAR100. 
To further verify the feasibility and effectiveness of our method in practical scenarios, we conduct experiments on a real-world dataset (\ie, Clothing1M)

You can download the CIFAR10 and CIFAR100 on [this](https://www.cs.toronto.edu/~kriz/cifar.html).

You can download the Clothing1M from [here](https://github.com/lightas/Occluded-DukeMTMC-Dataset).

# Training

Here is an example shell script to run CBS on CIFAR-10 :

```python
 python main.py --warmup-epoch 40 --epoch 250 --rho-range 0.6:0.6:100 --batch-size 128 --lr 0.05 --warmup-lr 0.01 --start-expand 200 --noise-type unif --closeset-ratio 0.4 --lr-decay cosine:40,5e-5,240  --opt sgd --dataset cifar10 --imbalance True --imb-factor 0.05 --alpha 0.6 --aph 0.35
```
# Results on Cifar10 and Cifar100

| Datasets               |  Cifar10               |   Cifar100                | 
|:-----------------------|:-----------------------|:--------------------------|
|  IF                    | [1,10,20,50,100,200]   |    [1,10,20]              |
|  NR                    |  [0.0,0.2,0.4]         |     [0.0,0.2,0.4,0.6]     |
|  CE                    |  74.49                 | 46.76                     |
|  Class-Balanced        |63.49                   |     42.81                 |
|  Focal                 |71.59                   |         43.85             |
|  LDAM-DRW              |  73.46                 |         45.47             |
|Co-teaching             |  60.63                 |         36.55             |
|O2U                     |  65.01                 |         40.21             |
|MW-Net                  |  74.13                 |         49.28             |
|HAR                     | 73.50                  |          42.88            |
|CurveNet                |  75.70                 | 50.49                     |
|Ours                    |78.47                   |     52.94                 |

