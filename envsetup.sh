conda create -n pytorch110-cuda113 python==3.8.11 -y

conda activate pytorch110-cuda113

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y

python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

conda install -c conda-forge kornia -y

conda install -c conda-forge einops -y

conda install -c conda-forge timm -y

pip install opencv-python

pip install pycocotools

conda install -c conda-forge tensorboard -y

pip install setuptools==59.5.0

pip3 install numpy==1.23.5

# conda deactivate

# conda create -n tensorboard python==3.8.11 -y

# conda activate tensorboard

# conda install -c conda-forge tensorboard -y

# conda deactivate

# conda activate pytorch110-cuda113 # Start training 