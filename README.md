# Research work for cloud and snow semantic segmentation problem using meteorological satellite Electro L2 multispectral data #

**ABSTRACT:**

This project is devoted to the method of clouds and snow detection according to the multispectral satellite images, received from a multizone scanning instrument used for hydrometeorological support and installed on the Russian satellite Electro-L No. 2. As the additional information, geographical information: latitude, longitude and altitude is used. The problem of snow and cloud discrimination is the absence of a spectral channel in the range 1400-1800 nm, which is necessary for their accurate separation, is considered. The results of this work include two new datasets from the meteorological satellites GOES-16, 17 and Electro-L No. 2 with the cloud and snow masks, as well as the trained Multi-Scale Attention Network (MANet) segmentation model, able to do accurate segmentation od snow and clouds for these satellites’ multispectral data. The proposed  neural network for clouds and snow segmentation has been tested for different seasons and daytime timestamps with different level of illumination of images. The developed algorithm is fully automatic, and it works in any season of the year during the daytime and is able to perform cloud and snow detection in real time mode for Electro-L No.2 data.

- Trained model for Electro L2 data can be loaded from model_for_Electro_L2 folder and used for data_inference in INFERENCE_PUBLIC.ipynb as an example

## Setup python version
The inference.ipynb file has been run with `python 3.9.7` on Windows 10 with NVIDIA CUDA supported 

#### All required libraries (follow requirements_inference.txt):

```bash
albumentations==1.3.0
h5py==3.8.0
ipython==8.12.0
matplotlib==3.7.1
netCDF4==1.6.4
numpy==1.23.5
opencv_python==4.7.0.68
opencv_python_headless==4.7.0.72
pandas==1.5.3
patchify==0.2.3
Pillow==9.5.0
pvlib==0.9.4
pyproj==3.4.1
PyWavelets==1.4.1
rasterio==1.2.10
Requests==2.31.0
scikit_learn==1.2.1
scipy==1.9.3
segmentation_models_pytorch==0.3.2
Shapely==1.8.4
tifffile==2021.7.2
torch==1.13.1+cu116
torcheval==0.0.6
torchmetrics==0.11.1
torchvision==0.14.1+cu116
tqdm==4.65.0
```
#### INFERENCE_PUBLIC.ipynb usage instructions:

- put all needed unarchieved files from inference_data folder and model_for_Electro_L2 folder and INFERENCE_PUBLIC.ipynb at the same directry
- Import all required packages and set up random seeds 42 everywhere.
- define your current directory and device: 
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
your_current_dir = os.getcwd()
```
- Than you can simply press 'run all' in your jupyter notebook and wait for parsing all inference data and working the MANet model predictiong through 192 pathes to eval cloud and snow masks for half off Electro L2 image
- Full precess of splitting channels to patches, making prediction and visualize it would take about 5 minutes, keep waiting
- During splitting, tqdm progress bar will help you to see the current progress stage 
- ATTENTION! Splitting all inference data will take about 4 GB of memory o your drive
- def merge_masks has 'save_mode' flag which is False by default, you can switch it to True to save the full prediction cloud and snow masks after merging it from patches 
- In the last cell their is a visualization of ground truth masks from GOES-16 satellite, MANet model prediction and RGB image from Electro L2 with the metrics (FAR, IoU, F1) evaluation

#### Acknowledgments:

Work is greatly supported by Non-commercial Foundation for the Advancement of Science and Education INTELLECT

