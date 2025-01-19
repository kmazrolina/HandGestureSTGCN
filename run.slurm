#!/bin/bash
#SBATCH --job-name=STGCN_Hand_training
#SBATCH --time=24:00:00
#SBATCH --account=plgrobot-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out

source ../venv/bin/activate

cache_path="/net/pr2/projects/plgrid/plggrai/kzrobek/MindEyeV2/.cache"
export XDG_CACHE_HOME=${cache_path}

jupyter nbconvert Train_STGCN.ipynb --to python

python Train_STGCN.py