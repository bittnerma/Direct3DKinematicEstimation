#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=long

# The default run (wall-clock) time is 1 minute
#SBATCH --time=7:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=16

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=16G

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once

# Use one gpu
#SBATCH --gres=gpu:turing:1

# Your job commands go below here

# Add path
source /tudelft.net/staff-bulk/ewi/insy/VisionLab/students/wyang/bash_profile
conda activate /tudelft.net/staff-bulk/ewi/insy/VisionLab/students/wyang/miniconda/miniconda

# Uncomment these lines when your job requires this software
module use /opt/insy/modulefiles
module load cuda/11.1 
module load cudnn/11.1-8.0.5.39 

# Complex or heavy commands should be started with 'srun' (see 'man srun' for more information)
# For example: srun python my_program.py
# Use this simple command to check that your sbatch settings are working (verify the resources allocated in the usage statistics)
free -m
nvidia-smi
srun python /tudelft.net/staff-bulk/ewi/insy/VisionLab/students/wyang/videoMuscle/videoMuscle/models/utils_h36m_metric_scale2.py --ymlFile /tudelft.net/staff-bulk/ewi/insy/VisionLab/students/wyang/env_cluster_next.yaml