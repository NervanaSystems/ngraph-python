# Deepspeech example

This example contains an implementation of Baidu SVAIL's [deep speech 2](https://arxiv.org/abs/1512.02595) model. It requires a functioning setup of [Warp-CTC](https://github.com/baidu-research/warp-ctc): [detailed here](https://github.com/NervanaSystems/ngraph/tree/master/third_party/warp_ctc).
To run this example, first install the ``python-Levenshtein`` package, which is used to calculate performance. Then, it's as easy as running:
```
python deepspeech.py
```

The script has a variety of configuration options. Use `python deepspeech.py --help` to check them out. This will download the training dataset from librispeech by default. If you would like to use a different dataset, first create an [aeon](https://github.com/NervanaSystems/aeon) dataloader manifest file for it, then use the ``--manifest_train`` option.
