# Camera Simulation for Learning

To-Do's

* Dataset Class
  - ~~Sample RAW processing~~
  - ~~Cleanup dirty variable names~~
  - Write documentations
* Sample Training with N3Net
* ~~Include Halide
* Write this readme

# Setup

Recommended: create a virtual environment.

'''
pip install -r requirements.txt
'''

Then build halide to enable halide modules.

# Note on buildding Halide

This repo submodules halide repos, so be sure to clone those.

Use `llvm-config-5.0` and `clang-5.0`. Set `LLVMCONFIG` and `CLANG` environment variables and just run `make -j8`. 

Make sure your CUDA is 8.0, and Pytorch 0.4.1. Other version may not work.

Don't clone from Halide from `jrk` or `mgharbi`, they include some updates that make it works with other version of pytorch that is not tested here.

You might have to do `pip install cffi`. I will include this in the requirements.txt soon.

# Testing That Pipeline Works

If you can run both jupyter notebooks without error, everything is good.

