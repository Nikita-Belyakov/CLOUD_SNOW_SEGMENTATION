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
```shelldevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
your_current_dir = os.getcwd()
```


