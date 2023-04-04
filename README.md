# Toronto3D entry for PyTorch
This is the PyTorch entry of Dataset [Toronto-3D](https://github.com/WeikaiTan/Toronto-3D). The preprocessing and loading patterns are the same as the [offical recommendations](https://github.com/WeikaiTan/RandLA-Net)
## Installation
### Requirements
- sklearn
- numpy
- pandas (optional)
- Cython
- PyTorch
### Compile CXX
> sh compile_op.sh
## Usage
### Preprocessing
1. downloads [Dataset](https://github.com/WeikaiTan/Toronto-3D#-download) and create a folder in Toronto-3D named original_ply
2. mv all .ply file to original_ply
3. change params in data_prepare_toronto3d.py according to your path and run it
### Dataset
after preprocessing, you can use toronto.py to provide a torch style dataset