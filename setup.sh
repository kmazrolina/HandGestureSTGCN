#!/bin/bash
#SBATCH --job-name=STGCN_Hand_setup
#SBATCH --time=10:00:00
#SBATCH --account=plgrobot-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu

## Setup Virtual Env
#python -m venv venv
source ../venv/bin/activate


# ##Install dependencies
# pip install \
# gdown \
# pandas \
# numpy \
# torch \
# torch_geometric \
# networkx \
# mediapipe \
# opencv-python \
# scikit-learn \
# matplotlib \
# jupyter


# ## Download Data
# mkdir data
# cd data

# #download video frames
# gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1GJ3ZzidMngAK5lbcLtX5aENfXelIFAQe
# mv frames/frames01.tgz frames/frames02.tgz frames/frames03.tgz frames/frames04.tgz frames/frames05.tgz .
# for file in *.tgz; do
#   tar -xvzf "$file"
# done
# rm frames01.tgz frames02.tgz frames03.tgz frames04.tgz frames05.tgz

# #download videos for inference only
# gdown --no-check-certificate https://drive.google.com/drive/folders/1GJ3ZzidMngAK5lbcLtX5aENfXelIFAQe
# tar -xvzf videos01.tgz


# #download annotations
# gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1-mihJEIFoNDpfo1puF8xAMJz6PGVKsBD

# #extract hand keypoints from video frames
# cd ..
python keypoint_detection.py 
