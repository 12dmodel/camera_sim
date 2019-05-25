# Camera Simulation for Learning

To-Do's

* Dataset Class
  - Write documentations

# Setup

Recommended: create a virtual environment.

```
pip install -r requirements.txt
```

Then build halide to enable halide modules. Read the notes in the following section first.

*Important:* This has only been tested with `PyTorch 0.4.1`. If you want to use Halide Modules, you must use this exact version of PyTorch. This is because other versions have different backend codes that breaks this particular version of Halide.

If you are training with [N3Net](https://github.com/visinf/n3net) (which we provide author's implementation), you will also need to install `pyinn` as follows:

```
pip install git+https://github.com/szagoruyko/pyinn.git@master
```

## Note on building Halide

This repo submodules halide repos, so be sure to clone those.

Use `llvm-config-5.0` and `clang-5.0`. Set `LLVMCONFIG` and `CLANG` environment variables and just run `make -j8`. We have seen some problem using other versions despite official instruction to use version 6.0 or later.

Make sure your CUDA is 8.0, and Pytorch 0.4.1. Other versions may not work. We have successfully compiled/run this code on both Ubuntu 14.04 and 16.04.

Don't clone from Halide from `jrk` or `mgharbi`, they include some updates that make it works with other version of pytorch that is not tested here.

Original instruction for compiling halide [here](https://github.com/12dmodel/gradient-halide), and [here](https://github.com/12dmodel/gradient_apps).

# Testing Camera Simulation Pipeline

If you can run `jupyter/pipeline_test.ipynb` and `jupyter/simulate_degradation.ipynb` notebooks without error, everything is good. The first one tests individual image processing modules, the second put them together in a pipeline and tries to process a raw iPhone 7 image.

# Training

We have include the training code based on authors' implementation of [N3Net](https://github.com/visinf/n3net). To setup: 
1. Download the clean patches which serves as the base of our dataset ([here](https://groups.csail.mit.edu/graphics/camera_sim/training_file_list.txt)).
2. Point the code to the folder by editing [this line](https://github.com/12dmodel/camsim/blob/master/dataset_specs/full_dataset.conf#L1).
3. Set a location to save checkpoints and log files by editing [this line](https://github.com/12dmodel/camsim/blob/master/denoiser_specs/full_dataset_n3net.conf#L2).
4. (Optional) edit [this convenience script](https://github.com/12dmodel/camsim/blob/master/tb.sh#L3) to run tensorboard to point to the checkpoint location

Then you should be able to run
```
python train.py --config_file=denoiser_specs/full_dataset_n3net.conf
```
And run tesorboard:
```
sh tb.sh full_dataset_n3net <Port>
```

# Testing Your Denoiser

The Jupyter Notebook `jupyter/real_benchmark.ipynb` contains benchmarking code that loads the denoiser and run it on the included sample patches. It will generate a webpage at a specified location. A version of our early-stopped, full-version dataset network is included in `samples/sample_output`.

We find that N3Net's memory consumption grows super-linearly with patch size. On a 12GB GPU, we were able to run at 200x200 patch max. If you run out of memory, you might need to reduce the patch size.

# Citation

If you use our code, please include this citation:

```
@article{jaroensri2019generating,
  title={Generating Training Data for Denoising Real RGB Images via Camera Pipeline Simulation},
  author={Jaroensri, Ronnachai and Biscarrat, Camille and Aittala, Miika and Durand, Fr{\'e}do},
  journal={arXiv preprint arXiv:1904.08825},
  year={2019}
}
```

# Acknowledgments

The authors would like to thank the Toyota Research Institute for their generous support of the projects. We thank
Tzu-Mao Li for his helpful comments, and Luke Anderson
for his help revising this draft
