# Character-level Convolutional Networks for Text Classification

This folder contains an implementation of the work of Zhang et al. (https://arxiv.org/abs/1509.01626) on classifying text data using character-level convolutional networks. To run the example, please follow the instructions in the section below to download the data and then run:
```
python char_conv.py --data_dir PATH_TO_DATASET --num_classes NUMBER_OF_CLASSES
```
For instance, in order to run the model on the DBPedia dataset, you need to download the 'dbpedia_csv' folder and then run:
```
python char_conv.py --data_dir dbpedia_csv --num_classes 14
```

## Datasets
We use the datasets prepared by the authors of the paper. They are available on the website of the official implementation [Crepe](https://github.com/zhangxiangxiao/Crepe). The datasets are hosted as tar archives, which decompress into folders containing a train.csv and a test.csv file.