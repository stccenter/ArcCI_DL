# ArcCI Deep Learning

###### Setting up files

1. Create a folder titled arcci-dl and then copy IceClassifier, setpy.py, and train_model.py into your newly created folder.
2. Create a folder titled resources inside of your arcci-dl folder. Within your resources folder, create three folders titled datasets, models, and logs.
3. Download the dataset [here](https://drive.google.com/drive/folders/1TDdFGWRyhGn7-ciDQX__if1jl-Rhmtpm?usp=sharing) and place the contents within your datasets folder.

###### Setting up your virtual environment
 
1. Create a conda environment with Python version 3.8 using ```conda create --name arcci-dl python=3.8```. If you do not have Anaconda, please download and install it from the [Anaconda site](https://www.anaconda.com/products/individual).
2. Ensure you have NVIDIA CUDA Toolkit version 10.2 on your system. If not, please download and install it from the [NVIDIA Developer site](https://developer.nvidia.com/cuda-toolkit).
3. Type ```conda activate arcci-dl```. You are now inside your virtual environment.
4. Run ```pip install -r requirements.txt``` to install all necessary packages and model dependencies.

###### Setting up Weights & Biases

1. Go to the [Weights & Biases site](https://wandb.ai/site) and create a Weights & Biases account and then a Weights & Biases project.
2. With your conda environment active, type ```wandb init``` and follow the prompt to enter your account credentials and select the project you created.
3. Run ```export WANDB_API_KEY=yourapikey``` with your Weights & Biases API key, which can be found in the settings section of your Weights & Biases account.

###### Running the code

1. Run ```python train_model.py --log_every_n_steps=5 --backbone_model=fcn_resnet50 --batch_size=60 --gpus=1 --learning_rate=2.1029136274973522e-05 --max_epochs=100 --optimizer=Adam```. Change the number of GPUs for distributed learning support (the default is one).
2. View your run and runtime metrics in Weights & Biases by navigating to the dashboard for the Weights & Biases project you created.