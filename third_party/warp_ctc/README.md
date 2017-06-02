# Bindings for Baidu's warp-ctc in Nervana Graph

Nervana Graph's CTC Op requires wrapping Baidu's [Warp-CTC](https://github.com/baidu-research/warp-ctc).
  
## Setup Instructions
1. Within an ngraph virtualenv, build Baidu's [Warp-CTC](https://github.com/baidu-research/warp-ctc) 
2. Update the environment variable LD_LIBRARY_PATH to point to the location of the built library (typically the location of the ``warp-ctc/build`` folder containing ``libwarpctc.so``).

