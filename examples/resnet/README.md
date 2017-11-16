# Deep Residual Networks
This example demonstrates training a deep residual network as first described in [He et. al.][msra1]. It can handle CIFAR10 and Imagenet datasets.

## Files
- *data.py*: Implements dataloader for CIFAR10 and imagenet datasets and creates aeon objects.
- *resnet.py*: Defines model for Residual network.
- *train_resnet.py*: Processes command line arguments, like the choice of dataset and number of layers, and trains the Resnet model.

## Dataset

The CIFAR10 Dataset gets downloaded automatically to *~/*. To download and use the dataset from a specific location, set ```--data_dir```.

The CIFAR100 Dataset gets downloaded automatically to *~/*. To download and use from a specific location, set ```--data_dir```.

### Imagenet

Download the dataset from http://image-net.org/download-images.

Intel® Nervana™ Graph uses the Aeon dataloader to efficiently generate macrobatches. Aeon requires manifests to be generated for the dataset. You only need to do this once using the following steps:

1. Set the environment variable: ``` export I1K_DATA_PATH=<Location to store manifests>```
2. Run this command: ```python ingest.py --input_dir <path to imagenet download location> --out_dir $I1K_DATA_PATH```
3. Run the following command:  ```export BASE_DATA_DIR=$I1K_DATA_PATH```
4. For imagenet update ```manifest_root``` to the location of manifests in *data.py*. 


## Usage
Use the following command to run training on Intel Nervana Graph:

```python train_resnet.py -b <cpu,gpu> --size <20,56> -t 64000 -z <64,128>```

## Citation
```
Deep Residual Learning for Image Recognition
http://arxiv.org/abs/1512.03385
```

[msra1]: <http://arxiv.org/abs/1512.03385>`
