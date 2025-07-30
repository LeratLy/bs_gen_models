# ModellingBrainStructureShapes

## Name
BrainStructureModelling

## Description
This project contains implementations of 3D Conditional Variational Autoencoders as well as Bernoulli Diffusion Autoencoders and Gaussian Autoencoders.
Additionally, an implementation of a 3D convolutional classifier ist added.

The implementations are based on https://github.com/takimailto/BerDiff and https://github.com/phizaz/diffae.
The models work on 3D binary segmentations masks and can be used to generate new data and explor data structure.

Model structure and code will be adapted for a broader usage.

## Visuals
Visuals with generated examples based on an open dataset will come soon.

## Installation
For the installation please use the given environment.yml as it contains all necessities to run the project. It contains a specifcation for an anaconda environment.
`conda env create -f environment.yml`
Packages can also be manually installed via pip.
To run the models you first have to define your base config. Previously used configs are saved in the model_templates file.

Add the following to your local `.git/config` for development to avoid any problems using git and jupyter notebook:
`[filter "strip-notebook-output"]
clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"`

Git lfs is used for folloing file types:
* `*.psd`

## Usage
An example with open source data will be added soon.

### Monitoring Training:
To monitor the training start the Tensorboard: 
+ When running test scripts: `tensorboard --logdir tests/runs`
+ When running from notebooks in main folder: `tensorboard --logdir ./runs`

For the case you are running the model on a remote server make sure that you forward the outputs to your local machine 
(`ssh -L 6006:127.0.0.1:6006 user@server`)

For the case you did not activate your conda environment run:
* `source ~/anaconda3/etc/profile.d/conda.sh`
* `conda activate <your_env_name>`

## Project structure 
* data: Folder that contain all your data and training csvs
* docs: Documentation with Sphinx
* models: You can save your model resutls here
* reports: Containing detailed results you can collect via training, is also used as plotting directory (if you want to use it add an empty figures folder)
* run_models: Contains run configurations and function for our model implementations (working ony with our checkpoints and data)
* src: Main project related files:
  * analysis: An analysis notebook and corresponding evaluation files, additional pipelines for preparing data for the evaluation, running evaluation and running the analysis. The analysis.ipynb is a playground for visualising and assessing results.
  * data: Data related scrips, such as for data loading and the creation of datasets.
  * models: Containing the main implementations of a CVAE, GDAE, and BDAE. All models inherit the same autoencoder base class and can be configured via a given `config`. Additionally, the fodler contains a `trainer.py` which is a trainer script that can be used to train validate and test your models. Each folder contains one model implementation.
  * utils: A folder containg all utility function used for training and data handling.
* tests: Folder containing test files for the project. Since a lot of the project structure changes, not all tests are up to date. Data laoding test rely on our dataset and may fail for others, as they assess size and dimensionality.
*  `environment.yml` file cotnaining specification of the anacodna environment.
*  `variables.py` main file where to hardcode project specific paths and patterns
## Authors and acknowledgment
The project is mainly based on the implementations of https://github.com/takimailto/BerDiff and https://github.com/phizaz/diffae, who made their code open source.
It is really appreciated.

## License
The project is licenced under the MIT license (please ifn information int the license file)

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

## Debugging
Duringt he project I ran into some environment issues because of version mismatches, following link helped fixing issues for me after reinstalling all dependencies:
https://stackoverflow.com/questions/76309946/conda-attributeerror-module-brotli-has-no-attribute-error-after-update
