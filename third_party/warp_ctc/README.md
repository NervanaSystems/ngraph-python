# Bindings for Baidu's warp-ctc in Nervana Graph

Nervana Graph's CTC Op requires wrapping Baidu's [Warp-CTC](https://github.com/baidu-research/warp-ctc).
  
## Setup Instructions
1. Within an ngraph virtualenv, run ```pip install cffi```.
2. Build Baidu's [Warp-CTC](https://github.com/baidu-research/warp-ctc) and set the environment variable WARP_CTC_PATH to point to the location of the built library (typically the location of the ``build`` folder containing ``libwarpctc.so``).

