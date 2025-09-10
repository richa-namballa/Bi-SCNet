#!/bin/bash
#SBATCH -c 2
#SBATCH --mem 8GB
#SBATCH --time 2:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=scnet_infer
#SBATCH --output logs/job.%J.scnet.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rn2214@nyu.edu
#SBATCH --account=pr_230_general

INPUT_DIR=/scratch/rn2214/source-separation/data/binaural_musdb18/test_mixtures
OUTPUT_DIR=/scratch/rn2214/source-separation/binaural-mss/data/output/scnet_binaural/test
CKPT_PATH=/scratch/rn2214/source-separation/SCNet/ckpt/binaural_checkpoint.th

../sing << EOF

conda activate scnet

python -m scnet.inference --input_dir $INPUT_DIR \
			  --output_dir $OUTPUT_DIR \
			  --checkpoint_path $CKPT_PATH

EOF
