# Research work for cloud and snow semantic segmentation problem using meteorological satellite Electro L2 multispectral data #

**Annotation:**

This project is devoted to the cloud and snow semantic segmentation using multispectral satellite images, received from a multizone scanning instrument (MSU-GS) used for hydrometeorological support and installed on the Russian satellite Electro-L No. 2. As the additional information, geographical information (latitude, longitude and altitude) is used. We povide the solution of the problem of snow and cloud discrimination is the absence of spectral channels in the range 1400-1800 nm, which are necessary for accurate separation snow from cloud textures. The results of this work include two new datasets from the meteorological satellites GOES-16, 17 and Electro-L No. 2 with the cloud and snow masks and geoinformation for each sample, as well as the trained Multi-Scale Attention Network (MANet) segmentation model, able to do accurate segmentation of snow and clouds for this satellites’ multispectral data. The proposed  neural network for clouds and snow segmentation has been tested for different seasons and daytime periods with different level of illumination of images. The developed algorithm is fully automatic, and it works in any season of the year during the daytime and is able to perform cloud and snow segmentation in real time mode for Electro-L No.2 and GOES-R data.

- Trained model for Electro L2 data can be loaded from model_for_Electro_L2 folder and used for data_inference in INFERENCE_PUBLIC.ipynb as an example
- Trained model for GOES-16,17 data can be loaded from model_for_GOES folder and used for GOES-R multispectral data from Amazon: https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/goes16_download.cgi?source=aws&satellite=noaa-goes16&domain=F&product=ABI-L2-MCMIP&date=2021-12-15&hour=12

## Setup python version
The inference_public.ipynb file has been run with `python 3.9.7` on Windows 10 OS with NVIDIA CUDA supported 

### All required libraries are written in requirements.txt
- There are seperated requrements.txt for each .ipynb in this repo! Be careful!
- Just run in your .ipynb this cell:
```
 !pip install -r requirements.txt
```
### INFERENCE_PUBLIC.ipynb usage instructions:

- Clone this repo on your PC and run INFERENCE_PUBLIC.ipynb
- Import all required packages and set up random seeds 42 everywhere.
- Define your current directory and device: 
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
your_current_dir = os.getcwd()
```
- Its recomended to use CUDA device instead of CPU to make the predicting process of model faster
- After that you can simply press 'run all' in your jupyter notebook and wait for parsing all inference data and working the MANet model predictiong through 1152 patches to eval cloud and snow masks for half of Electro L2 image
- Its recomended to use local PC for running def merge_masks as Google Collab slowly works with opening lots of files from your Google Drive content directories in a loop
- The complete process of splitting channels to patches, making prediction and visualization can take **more than 5 minutes, keep waiting, please**
- During splitting, `tqdm` progress bar will help you to see the current progress stage 
- **ATTENTION!** Splitting all inference data requires about **2 GB** of free memory on your drive
- `def merge_masks` has 'save_mode' flag which is False by default, you can switch it to `True` to save the full prediction cloud and snow masks after merging it from patches
- During loading model there is a flag `GOES_mode` which is False by default, you can switch it to `True` to use weights of MANet model, trained on GOES data
- In the last cell there is a visualization of ground truth masks from GOES-16 satellite, MANet model prediction and RGB image from Electro L2 with the metrics (FAR, IoU, F1) evaluation

### GOES_data folder usage instructions:

- There is PARSING_&_PROCESSING_SATELLITE_DATA_PUBLIC.ipynb with lots of utils and very useful functions for GOES-16,17 data and it can be easily adapted for Electro L1, L2, L3, L4, GOES-18 or SEVIRI Meteosat-9 satellites as well
- This notebook can help you to open and process all needed GOES-R multispectral data and Level2 (L2) products to make your own dataset with snow and cloud masks
- PARSING_&_PROCESSING_SATELLITE_DATA_PUBLIC.ipynb includes cells for reprojecting Aster GDEM and Snow map 2D images from plate caree projection to geostationary projetion according needed satellite selected
- All needed multispectral data from GOES-R can be downloaded for free from here: https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/goes16_download.cgi?source=aws&satellite=noaa-goes16&domain=F&product=ABI-L2-MCMIP&date=2021-12-15&hour=12
- It's recommended to download and process Cloud & Moisture L2 product in multiband format with 2 km resolution as it's easier and faster to parse
- All needed snow data from MODIS can be downloaded for free from 2 sources (there also all required cells for evaluate binary snow mask from both these formats provided in PARSING_&_PROCESSING_SATELLITE_DATA_PUBLIC.ipynb):
  - https://n5eil01u.ecs.nsidc.org/MOST/MOD10C1.061/2023.05.15/
  - https://neo.gsfc.nasa.gov/view.php?datasetId=MOD10C1_E_SNOW&date=2023-02-15

## Acknowledgments:

**Work is greatly supported by Non-commercial Foundation for the Advancement of Science and Education INTELLECT**

