# Good Semi-supervised VAE Requires Tighter Evidence Lower Bound

Here is the official implementation of the model `OSPOT-VAE` in paper ["Good Semi-supervised VAE Requires Tighter Evidence“ Lower Bound"](https://openreview.net/forum?id=S1ejj64YvS).

## Model Review

* **One-stage SSL VAE**

	![one-stage-vae](./image/one-stage-schematic.svg)

* **Optimal Transport Estimation**

	![optimal transport estimation](./image/optimal-transport-schematic.svg)

# Setup

### Install Package Dependencies

```
Python Environment: >= 3.6
torch = 1.2.0
torchvision = 0.4.0
scikit-learn >= 0.2
tensorbard >= 2.0.0
```

### Install Datasets

We need users to declare a `base path` to store the dataset as well as the log of training procedure. The  *directory structure* should be

```yaml
base_path
│       
└───dataset
│   │   cifar
│       │   cifar-10-batches-py
│       │   |	...
│       │   cifar-100-python
│       │   |	...
│   │   svhn
│       │   ...
│   │   mnist
│       │   ...
└───trained_model_1
│   │	parmater
│   │	runs
└───trained_model_2
│   │	parmater
│   │	runs
...
└───trained_model_n
│   │	parmater
│   │	runs
```

We refer users to use the following functions in `torchvision` to install datasets

```shell
from os import path
import torch
import torchvision
# set base_path
base_path = "./"
# install mnist,svhn,cifar10,cifar100
torchvision.datasets.MNIST(path.join(base_path,"dataset","mnist"),download=True)
torchvision.datasets.CIFAR10(path.join(base_path,"dataset","cifar"),download=True)
torchvision.datasets.CIFAR100(path.join(base_path,"dataset","cifar"),download=True)
torchvision.datasets.SVHN(path.join(base_path,"dataset","cifar"),download=True)
```

Or you can manually put the dataset in the appropriate folder.

## Running

Notice that we have implemented 3 categories of backbones: **WideResNet, PreActResNet and Densenet**. Here we give the example for **WideResNet**. To use other network backbone, please change `--net-name` parameter (e.g. `--net-name preactresnet18`).

For **CUDA computation**, please set the `--gpu` parameter (e.g. `--gpu "0,1"` means to use gpu0 and gpu1 together to do calculation).

### Semi-supervised Learning

#### OSPOT-VAE in Cifar10 (4k) and Cifar100 (4k and 10k) [Table. 2,3]

Here we list several important parameters **need to be set manually** in the following table

| Parameter       | Means                                                        |
| --------------- | ------------------------------------------------------------ |
| kbmc            | The max kl beta for continuous latent variable $z$.          |
| akb             | The epoch to adjust kl beta (e.g. akb=200 means the kl beta for $z$ and $c$ will get their max value at the 200th epoch). |
| cmi             | Mutual information for continuous latent variable $z$        |
| dmi             | Mutual information for discrete latent variable $c$          |
| apw             | The epoch to adjust posterior weight $w_{R_{z}}$             |
| br              | If we use BCE loss in $p_{\theta}(x;z,c)$, default is False. |
| annotated_ratio | The annotated ratio for dataset.                             |
| ad              | The milestone list for adjust learning rate.                 |
| epochs          | The total epochs in training process                         |

1. For Cifar10 (4k), please use the following command

   ```shell
   # for wideresnet-28-2
   python main_ospot_vae.py -bp basepath --net-name wideresnet-28-2 --gpu gpuid --br
   # for wideresnet-28-10
   python main_ospot_vae.py -bp basepath --net-name wideresnet-28-10 --gpu gpuid --br
   ```

2. For Cifar100 (4k), please use the following command

   ```shell
   # for wideresnet-28-2
   python main_ospot_vae.py -bp basepath --net-name wideresnet-28-2 --gpu gpuid --dataset "Cifar100" --akb 150 --apw 400 --annotated-ratio 0.1 --cmi 1280 --dmi 4.6 --kbmc 0.1 -ad [500,600,650] --epochs 700 --br
   # for wideresnet-28-10
   python main_ospot_vae.py -bp basepath --net-name wideresnet-28-10 --gpu gpuid --dataset "Cifar100" --akb 150 --apw 400 --annotated-ratio 0.1 --cmi 1280 --dmi 4.6 --kbmc 0.1 -ad [500,600,650] --epochs 700
   ```

3. For Cifar100 (10k), please use the following command

   ```shell
   # for wideresnet-28-2
   python main_ospot_vae.py -bp basepath --net-name wideresnet-28-2 --gpu gpuid --dataset "Cifar100" --akb 150 --apw 400 --annotated-ratio 0.25 --cmi 1280 --kbmc 0.1 -ad [500,600,650] --epochs 700 --br
   # for wideresnet-28-10
   python main_ospot_vae.py -bp basepath --net-name wideresnet-28-10 --gpu gpuid --dataset "Cifar100" --akb 150 --apw 400 --annotated-ratio 0.25 --cmi 1280 --kbmc 0.1 -ad [500,600,650] --epochs 700
   ```

*The performance of test dataset in training process for different dataset is listed as:*

* **Cifar10 (4k)**

  * *WideResNet-28-2*

  * ![Cifar10-4K-WRN-28-2](./image/Cifar10-4K-WRN-28-2.png)

  * *WideResNet-28-10*

    ![Cifar10-4K-WRN-28-10](./image/Cifar10-4K-WRN-28-10.png)

* **Cifar100 (4k)**
	
  * *WideResNet-28-2*

	  ![Cifar100-4k-WRN-28-2](./image/Cifar100-4k-WRN-28-2.png)
	
	* *WideResNet-28-10*
	![Cifar100-4k-WRN-28-10](./image/Cifar100-4k-WRN-28-10.png)
	
* **Cifar100 (10k)**

  * *WideResNet-28-2*

    ![Cifar100-WRN-28-2](./image/Cifar100-WRN-28-2.png)

  * *WideResNet-28-10*

	  ![Cifar100-WRN-28-10](./image/Cifar100-WRN-28-10.png)

#### One-stage SSL  VAE in MNIST (100) and SVHN (1k) [Table.1]
Use the following commands to reproduce our results
```
# run One-stage SSL VAE on MNIST (100)
python main_one_stage_vae_mnist.py -bp basepath --gpu gpuid

# run One-stage SSL VAE on SVHN (1k)
python main_one_stage_vae_svhn.py -bp basepath --gpu gpuid
```


### Ablation Study [Table.4]

* **Optimal Transport Estimation combines Encoder**

  Here we use optimal transport estimation only for the representation of classification $q_{\phi}(c\vert x)$, our implementation is similar to the [Manifold Mixup Method](https://pdfs.semanticscholar.org/d4c7/b0886e4033ab8f56c0edea3a27d0bfe46c1b.pdf)[1]. 

  Use the following commands to reproduce our results

  ```shell
  # cifar10 (4k)
  python main_optimal_transport_estimation.py -bp basepath --gpu gpuid 
  # cifar100 (10k)
  python main_optimal_transport_estimation.py -bp basepath --gpu gpuid --dataset "Cifar100" --ar 0.25
  # SVHN (1k)
  python main_optimal_transport_estimation.py -bp basepath --gpu gpuid --dataset "SVHN"
  ```

### Generative Performance [Table.5]

* **Pure VAE**

  Here we only use ELBO to train a pure VAE with the following commands

  ```shell
  # cifar 10
  python main_pure_vae.py --cmi 200 --dmi 2.3 -bp basepath --gpu gpuid 
  # cifar 100
  python main_pure_vae.py --cmi 1280 --dmi 4.6 --ldd 100 -bp basepath --gpu gpuid 
  ```

* **Generated Examples of OSPOT-VAE**

  * *MNIST*

    ![mnist](./image/mnist.png)

  * *SVHN*

    ![svhn](./image/svhn.png)

  * *Cifar10*

    ![cifar10_ssl](./image/cifar10_ssl.png)

  * *Cifar100*

    ![cifar100_ssl](./image/cifar100_ssl.png)

## Reference

[1] Verma, Vikas, et al. "Manifold mixup: Encouraging meaningful on-manifold interpolation as a regularizer." *stat* 1050 (2018): 13.

