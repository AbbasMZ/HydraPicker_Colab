# HydraPicker

HydraPicker is a learning-based approach for picking particle in Cryo-EM micrographs.
It considers bias towards multiple structures to achieve an improved generalized model and a specialized model for each of the training structures.

For more details about 'HydraPicker' please refer to this paper:

Masoumzadeh, A. and Brubaker, M., HydraPicker: Fully Automated Particle Picking in Cryo-EM by Utilizing Dataset Bias in Single Shot Detection, BMVC 2019.\
[Paper](https://bmvc2019.org/wp-content/uploads/papers/1044-paper.pdf)\
[Supplimentary material](https://bmvc2019.org/wp-content/uploads/papers/1044-supplementary.zip)

## Installation

HydraPicker is only tested using Manjaro and Arch distribution of Linux, and requires PyTorch V2, Python V3.6.5, and a customized version of FastAI. A NVIDIA GPU with at least 8GB VRAM is required for most functionalities of HydraPicker.
If you plan to further increase the number of structures, even larger VRAM will be required.

Make sure of the compatibility of your NVIDIA driver version with CUDA 9 or CUDA 10.

It's highly recommended that you install HydraPicker and its dependencies in a virtual environment (conda or others), to avoid interference with system-wide python packages.

Create a new Conda Environment and install the required packages using one of the .yml files according to supported CUDA version for your system.

`conda env create -f env_CUDA_9.yml`\
or\
`conda env create -f env_CUDA_10.yml`

Download the content of this repository to the target location:

`git clone https://github.com/AbbasMZ/HydraPicker`\
`cd HydraPicker`

Download the generalized model and move it to the installation directory in `data\models`.

[Link to the model (Size is about 1 GB.)]()

To make sure the setup is working correctly, try using the model to pick from a sample dataset 'Empiar 10078':

`CUDA_VISIBLE_DEVICES=0 python predict.py -td 10078 -l SSPicker_29_6_4b_00945best -psf test`

Asuming no error would occur three .star files should be created in `predictions/test`.

## Usage

To make sure the setup is working correctly

The provided model is trained on 37 datasets provided by [Warp](https://www.nature.com/articles/s41592-019-0580-y).

## License

Please find the related information in 'LICENSE' and 'NOTICE' files.