# conda create --name AOA-track python=3.7 -y
# conda activate AOA-track

# CUDA 10.1
conda install cudatoolkit=10.1
pip install git+https://github.com/TAO-Dataset/tao
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install onnx==1.7.0 onnxruntime-gpu==1.4.0 opencv-python==4.0.1.24 scikit-learn==0.22.1


# install other dependencies
pip install Pillow

# we cant use a version protobuf that is too new (TypeError in gen_detection_features)
pip install protobuf==3.20.

