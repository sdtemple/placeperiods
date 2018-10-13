#!/bin/bash
#SBATCH --partition=gpu        
#SBATCH --job-name=pptrain  
#SBATCH --output=pptrain.out 
#SBATCH --error=pptrain.err   
#SBATCH --time=1-00:00:00         
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

module load anaconda3/5.1.0
conda create --name pyfull python=3.6 --yes
source activate pyfull
conda install keras-gpu --yes
pip install --upgrade gensim
conda install matplotlib --yes
conda install scikit-learn --yes
conda install nltk --yes
python -u /projects/fickaslab/stemple/pythons/pptrain.py $SLURM_JOB_GPUS
source deactivate
conda env remove --name pyfull --yes