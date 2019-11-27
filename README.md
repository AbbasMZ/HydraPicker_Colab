# HydraPicker

HydraPicker is a learning-based approach for picking particle in Cryo-EM micrographs.
It considers bias towards multiple structures to achieve an improved generalized model and a specialized model for each of the training structures.

## Installation

This program uses PyTorch Deep Learning Framework and a customized version of FastAI.
Create a new Conda Environment and install the required packages using the environment.yml file.

`conda env create -f environment.yml`

Please download the generalized model and move it to the put in the installation directory in the following path:

`data\models`

[Link to the model (Size is about 1 GB.)]()

## Model



## Usage


The provided model is trained on 37 datasets provided by [Warp](https://www.nature.com/articles/s41592-019-0580-y).

## Reference

For more details about 'HydraPicker' please refer to this paper:

Masoumzadeh, A. and Brubaker, M., HydraPicker: Fully Automated Particle Picking in Cryo-EM by Utilizing Dataset Bias in Single Shot Detection, BMVC 2019.\
[Paper](https://bmvc2019.org/wp-content/uploads/papers/1044-paper.pdf)\
[Supplimentary material](https://bmvc2019.org/wp-content/uploads/papers/1044-supplementary.zip)

## License

Please find the related information in 'LICENSE' and 'NOTICE' files.