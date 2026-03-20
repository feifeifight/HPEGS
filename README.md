# HPEGS: Unified Hash Physics-Informed Neural Network and Entropy Regularization in Gaussian Splatting for sparse view synthesis

This is the official repository for our ECCV 2026 paper **HPEGS: Unified Hash Physics-Informed Neural Network and Entropy Regularization in Gaussian Splatting for sparse view synthesis**.

![image](assets/HPEGS.jpg)


## Installation

Tested on Ubuntu 22.04, CUDA 11.3, PyTorch 1.12.1

``````
conda env create --file environment.yml
conda activate hpegs

cd submodules
git clone git@github.com:ashawkey/diff-gaussian-rasterization.git --recursive
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./diff-gaussian-rasterization ./simple-knn
``````

If encountering installation problem of the `diff-gaussian-rasterization` or `gridencoder`, you may get some help from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [torch-ngp](https://github.com/ashawkey/torch-ngp).


For reference, the best results we get in two random tests are as follows:

| PSNR   | LPIPS  | SSIM  |
| ------ | ------ | ----- | 
| 22.272 | 0.087  | 0.901 | 





## Acknowledgement

This code is developed on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) with [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) and a modified [diff-gaussian-rasterization](https://github.com/ashawkey/diff-gaussian-rasterization). The implementation of neural renderer are based on [torch-ngp](https://github.com/ashawkey/torch-ngp). Codes about [DPT](https://github.com/isl-org/MiDaS) are partial from [SparseNeRF](https://github.com/Wanggcong/SparseNeRF). Thanks for these great projects!
