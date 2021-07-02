# model_convert

#### Description
{**When you're done, you can delete the content in this README and update the file with details for others getting started with your repository**}

#### Software Architecture
Software architecture description

#### Installation

bellow is the installation instruction for configuration of CUDA 11.1 + cudnn 8.0 + TensorRT 7.2.2.3 for cuda11.1 cudnn 8.0  + conda with python 3.8  (lj_torch)+ pytorch1.8.1 for cuda11.1.
You can adopt other configurations and accordingly modify the version name in the following commands.

(1) CUDA 11.1 + cudnn 8.0
sudo ln -s cuda-11.1 cuda  # for all users, do this only once
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
export CUDA_DEVICE=0      # may be needed for some codes

(2) TensorRT 7.2.2.3 for cuda11.1 cudnn 8.0 
# take care to select the TensorRT version, make sure TensorRT must assist your CUDA and cuddnn version, which you can find out in the TensorRT install file name.
tar -xzvf TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz  (for all users, do this only once)
sudo cp -r TensorRT-7.2.2.3 /usr/local # for all users, do this only once
export PATH=/usr/local/TensorRT-7.2.2.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/TensorRT-7.2.2.3/lib:$LD_LIBRARY_PATH

cd /usr/local/TensorRT-7.2.2.3/samples/trtexec
sudo make clean & sudo make   # for all users, do this only once

(3) conda environment(python3.8 + pytorch1.8.1 for cuda11.1 + TensorRT-7.2.2.3 for python)
conda create -n lj_torch python=3.8
conda activate lj_torch
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
cd /usr/local/TensorRT-7.2.2.3/python
pip3 install tensorrt-7.2.2.3-cp38-none-linux_x86_64.whl  # copy the .whl file to a folder needs no sudo
cd /usr/local/TensorRT-7.2.2.3/graphsurgeon
pip3 install graphsurgeon-0.4.5-py2.py3-none-any.whl # copy the .whl file to a folder needs no sudo
cd /usr/local/TensorRT-7.2.2.3/onnx_graphsurgeon
pip3 install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl # copy the .whl file to a folder needs no sudo
pip3 install pycuda

#### Instructions
1. (Optional) create an onnx model to test, run the follong code in python in your prepared conda enrionment:
    import torchvision.models as models
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    import torch
    BATCH_SIZE = 64
    dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)
    import torch.onnx
    torch.onnx.export(resnext50_32x4d, dummy_input, "test.onnx", verbose=False)

2. (Optional) use trtexec to compare with this project, or to check if TensorRT installed correctly, e.g.
[TensorRT install path]/trtexec --onnx=test.onnx --saveEngine=test.trt --explicitBatch --verbose

3. do the test and make sure no error appear
python test_model_convertor.py  (before running, modify paths in this script)
python test_mmdet_ssd.py (before running, modify paths in this script)

4. (Optional) to test your other models, modify in load_test_model in file test_model_convertor.py

5. call ModelConvertor from your codes

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
