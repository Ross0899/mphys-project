# MPhys Project Code-Base
## Analysing Nanoparticle Size and Shape Using Image Analysis and Machine Learning

The conda environment file is named requirements.txt and can be installed using 

```bash
conda create --name <env> --file requirements.txt
```


* __/cnn__ contains the training and prediction scripts for the convolutional neural network (CNN) models

* __/generate_particles__ contains the Python scripts for generating images of particles. It also contains scripts to divide, augment and overlay the images/masks. size_distribution.py takes a CNN model and uses it to predict a size distribution for unseen synthetic data.

* __/random_forest__ contains the training and prediction random forest scripts. It also has scripts which crop and divide images.

