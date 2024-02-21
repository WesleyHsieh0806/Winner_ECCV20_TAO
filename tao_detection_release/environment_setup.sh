# conda create --name AOA python=3.7 -y
# conda activate AOA



# CUDA 10.1
pip install torch==1.4.0 torchvision==0.5.0

pip install cython
pip install numpy
pip install six
pip install pycocotools
pip install mmcv==0.2.14
pip install matplotlib
pip install terminaltables
pip install imagecorruptions
pip install albumentations
pip install tqdm
pip install addict

cd lvis-api/
python setup.py develop

cd ..
python setup.py develop