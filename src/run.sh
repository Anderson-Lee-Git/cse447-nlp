#!/bin/bash
#
#SBATCH --job-name=student-train
#SBATCH --account=stf
#SBATCH --partition=ckpt
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --constraint="[rtx6k|a40|a100]"
#SBATCH --requeue
#
#SBATCH --open-mode=truncate
#SBATCH --chdir=/gscratch/scrubbed/lee0618/cse447-nlp/src
#SBATCH --output=/gscratch/scrubbed/lee0618/cse447-nlp/src/log/out.log
#SBATCH --error=/gscratch/scrubbed/lee0618/cse447-nlp/src/log/out.err

export PATH=$PATH:$HOME/miniconda3/bin

echo "---------start-----------"

echo "Train student model on filtered text"
python train_student_model.py --model_name "gpt2-medium"
python train_student_model.py --model_name "roberta-base"