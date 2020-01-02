#One-stage VAE

The implementation of One-stage VAE for semi-supervised learning on MNIST and SVHN. Code base credit for [JointVAE](https://arxiv.org/abs/1804.00104). For setup details, please refer to the documentation of OSPOT-VAE.

##Usage
    # run semi-supervised learning on MNIST (100)
    python mnist_trainer.py 
    
    # run semi-supervised learning on SVHN (1k)
    python svhn_trainer.py 
    
###Arguments to be specified yourself manually

| Arguments | Descriptions |
|:---:|:---:|
| --save-dir | directory location where you wanna save your training logs and models |
|--path-to-data| directory location where your raw data is stored |
|--gpu| GPU id your wanna use |
|--train-time| the x-th time of training |

Other arguments are with default values.


