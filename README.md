The flash attention v2 kernel has been extracted from [the original repo](https://github.com/Dao-AILab/flash-attention) into this repo to make it easier to integrate into a third-party project. In particular, the dependency on libtorch was removed.

As a consquence, dropout is not supported (since the original code uses randomness provided by libtorch). Also, only forward is supported for now.
```
git clone --recursive https://github.com/JakeFlasher/libflash_attn.git
```

Build with (make sure to check if cmake env variables of CUDA is correctly set!)
```
mkdir build && cd build
cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.0 ..
make
```

It seems there are compilation issues if g++-9 is used as the host compiler. We confirmed that g++-11 works without issues.
