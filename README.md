# Brain Structure Generative Models
## Description
This project contains implementations of 3D Conditional Variational Autoencoders as well as Bernoulli Diffusion Autoencoders and Gaussian Diffusion Autoencoders.
Additionally, an implementation of a 3D convolutional classifier is given.
The models are used to generate 3D segmentation masks of the Choroid Plexus based on different MS types, but all models can be applied to other scenarios as well and can be used to generate new data and explor underlying data structures.

The implementation if the bernoulli diffusion process is based on https://github.com/takimailto/BerDiff and the diffusion autoencoder is based on https://github.com/phizaz/diffae.

## Visuals
Visuals with generated examples based on an open dataset will come soon.

## Installation

### Git LFS
The repository contains 3D segmentation masks and trained models, which are added using git lfs.
If not done already, please install git lfs. A useful manual can be found here: https://josh-ops.com/posts/add-files-to-git-lfs/.
Git lfs should be configured for following file types: `*.psd` `*.pkl` `*.npz` `*.pt`.
To add single files use `git lfs track --filename [file path]`.

### Python Packages
You need a python package manager like anaconda to create an environment or use a virtual environment with pip.
For the installation please use the given `environment.yml` as it contains all necessities to run the project. <br><br>
**Before installing the packages**, make sure that the pytorch cuda version suits your cuda version or use the cpu pytorch package (this is the default option).
If you use cuda, please refer to https://pytorch.org/get-started/locally/ for instructions on how to get the right version.
Then update version details in the environment yml e.g.:
```
- torch --index-url=https://download.pytorch.org/whl/cu121 <-- your cuda version
- torchvision --index-url=https://download.pytorch.org/whl/cu121 <-- your cuda version
``` 

Then install all packages as defined in the environment file:
```
conda env create -f environment.yml
```
Then you can activate your conda environment and start working with the project:
``
conda activate mec_brain
``
If you need to lik conda before, use `source ~/anaconda3/etc/profile.d/conda.sh`. 
Alternatively, you can install the listed packages via pip in a virtual environment without using the environment file.

### Git and Jupyter Notebook
Add the following to your local `.git/config` for development to avoid any problems using git and jupyter notebook:
```
[filter "strip-notebook-output"]
clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```

Now you are ready to work with the project.

## Usage
An example with open source data will be added soon.

### Monitoring Training:
To monitor the training start the Tensorboard: 
+ When running test scripts: `tensorboard --logdir tests/runs`
+ When running from notebooks in main folder: `tensorboard --logdir ./runs`

For the case you are running the model on a remote server make sure that you forward the outputs to your local machine 
(`ssh -L 6006:127.0.0.1:6006 user@server`)

## Project structure 
* **data**: Folder that contain all your data and training csvs
* **docs**: Documentation with Sphinx
* **models**: You can save your model resutls here
* **reports**: Containing detailed results you can collect via training, is also used as plotting directory (if you want to use it add an empty figures folder)
* **run_models**: Contains run configurations and function for our model implementations (working ony with our checkpoints and data)
* **src**: Main project related files:
  * **analysis**: An analysis notebook and corresponding evaluation files, additional pipelines for preparing data for the evaluation, running evaluation and running the analysis. The analysis.ipynb is a playground for visualising and assessing results.
  * **data**: Data related scrips, such as for data loading and the creation of datasets.
  * **models**: Containing the main implementations of a CVAE, GDAE, and BDAE. All models inherit the same autoencoder base class and can be configured via a given `config`. Additionally, the fodler contains a `trainer.py` which is a trainer script that can be used to train validate and test your models. Each folder contains one model implementation.
  * **utils**: A folder containg all utility function used for training and data handling.
* **tests**: Folder containing test files for the project. Since a lot of the project structure changes, not all tests are up to date. Data laoding test rely on our dataset and may fail for others, as they assess size and dimensionality.
*  `environment.yml` file cotnaining specification of the anacodna environment.
*  `variables.py` main file where to hardcode project specific paths and patterns
## Authors and acknowledgment
The project is mainly based on the implementations of https://github.com/takimailto/BerDiff and https://github.com/phizaz/diffae, who made their code open source.
It is really appreciated.

## License
The project is licenced under the MIT license (further information in license file)

## Project status
As the project was tailored to a specifc use case, all functionalities might not work out of the box in a new project setup. Please be aware of this. Modifications and tests with open source data and models, making it broadly applicable will follow.

## Debugging
During the project I ran into some environment issues because of version mismatches, following link helped fixing issues after reinstalling all dependencies:
https://stackoverflow.com/questions/76309946/conda-attributeerror-module-brotli-has-no-attribute-error-after-update
