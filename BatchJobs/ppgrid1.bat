#!/bin/bash
#SBATCH --partition=gpu           
#SBATCH --job-name=ppgrid1    
#SBATCH --output=ppgrid1.out  
#SBATCH --error=ppgrid1.err   
#SBATCH --time=1-00:00:00         
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load anaconda3/5.1.0
conda create --name pytemp1 python=3.6 --yes
source activate pytemp1
conda install keras-gpu --yes
pip install --upgrade gensim
conda install matplotlib --yes
conda install scikit-learn --yes
conda install nltk --yes
python -u /projects/fickaslab/stemple/pythons/pp_grid1.py $SLURM_JOB_GPUS
source deactivate
conda env remove --name pytemp1 --yes