# ArcCI Deep Learning Module

The project implementation supports the following:
- Single GPU implementation
- Multiple GPUs for distributed learning

## **Setup**

###### Set up files

1. Create a folder titled arcci-dl and then copy IceClassifier, setpy.py, requirements.txt, and train_model.py into your newly created folder.
2. Create a folder titled resources inside of your arcci-dl folder. Within your resources folder, create three folders titled datasets, models, and logs.
3. Within your datasets folder, create a folder titled ice-tiles-dataset-badlight.
4. Download the dataset [here](https://drive.google.com/drive/folders/1mGczHAOYH0Vxe5ZK2yynDdqnmvSpuf1c) and place the 256x256 folder within your ice-tiles-dataset-badlight folder.

###### Set up your virtual environment

1. Create a conda environment with Python version 3.8 using ```conda create --name arcci-dl python=3.8```. If you do not have Anaconda, please download and install it from the [Anaconda site](https://www.anaconda.com/products/individual).
2. Ensure you have NVIDIA CUDA Toolkit version 11.1 on your system. If not, please download and install it from the [NVIDIA Developer site](https://developer.nvidia.com/cuda-toolkit).
3. Type ```conda activate arcci-dl```. You are now inside your virtual environment.
4. Run ```pip install -r requirements.txt``` to install all necessary packages and model dependencies.

## **Run the model**

1. Run ```python train_model.py --log_every_n_steps=5 --backbone_model=fcn_resnet50 --batch_size=60 --gpus=1 --learning_rate=2.1029136274973522e-05 --max_epochs=100 --optimizer=Adam --local_mode=True --dataset_path=/.../resources/datasets/ice-tiles-dataset-badlight/256x256```. Change the number of GPUs for distributed learning support (the default is one). Change the dataset_path to your own file path (you can use the ```pwd``` command inside of the 256x256 directory to find it). If you encounter ```RuntimeError: CUDA out of memory```, please decrease the batch size.
