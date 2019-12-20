# HydraPicker

HydraPicker is a learning-based approach for picking particle in Cryo-EM micrographs.
It considers bias towards multiple structures to achieve an improved generalized model and a specialized model for each of the training structures.

For more details about 'HydraPicker' please refer to the paper:

Masoumzadeh, A. and Brubaker, M., HydraPicker: Fully Automated Particle Picking in Cryo-EM by Utilizing Dataset Bias in Single Shot Detection, BMVC 2019.\
[Paper](https://bmvc2019.org/wp-content/uploads/papers/1044-paper.pdf)\
[Supplimentary material](https://bmvc2019.org/wp-content/uploads/papers/1044-supplementary.zip)

## Setup

This version is meant to run on [Google Colab](https://colab.research.google.com).
It is an online jupyter notebook that gives you access to a GPU and has some common python deep learning libraries pre-installed.

If you have not used used Colab before, first setup your Google Colab account so that you would have a `Colab Notebooks` folder in your [Google Drive](https://drive.google.com).
Then, clone the contents of this repository to your computer and then upload to your Google Drive under `Colab Notebooks`.
Make sure you have a few Gigs of free space on your Google Drive.

First download this pre-trained model and upload it to your Google Drive under `Colab Notebooks\HydraPicker_Colab\data\models`.

[Link to the model (Size is about 1 GB.)](https://www.dropbox.com/sh/eqa3oh0gprdl518/AADyZl0zPjCsqJyYwMf09Jq1a?dl=0)

Then, follow the instructions in `Colab Notebooks\HydraPicker_Colab\colab_sample_run.ipynb` to test that everything is working.

## Usage

There are 3 main scripts: train, predict, and evaluate.
Please refer to each script for more details on their options.

## License

Please find the related information in 'LICENSE' and 'NOTICE' files.
Please also note that the provided model is trained on 37 datasets provided by [Warp](https://www.nature.com/articles/s41592-019-0580-y).