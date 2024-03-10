#!/bin/bash
#
#SBATCH --job-name=student-train
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --requeue
#
#SBATCH --open-mode=truncate
#SBATCH --chdir=/gscratch/scrubbed/lee0618/cse447-nlp/src
#SBATCH --output=/gscratch/scrubbed/lee0618/cse447-nlp/src/log/out.log
#SBATCH --error=/gscratch/scrubbed/lee0618/cse447-nlp/src/log/out.err

export PATH=$PATH:$HOME/miniconda3/bin

echo "---------start-----------"

echo "Teacher model generates raw text"
python teacher_generate.py

echo "Critique model filters generated text"
python critique_filter.py

echo "Train student model on filtered text"
python train_student_model.py