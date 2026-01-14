#!/bin/bash
#SBATCH -c 24
#SBATCH --mem 128GB
#SBATCH --time 47:59:00
#SBATCH --gres gpu:rtx8000:4
#SBATCH --job-name=scnet_train
#SBATCH --output logs/job.%J.scnet.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rn2214@nyu.edu
#SBATCH --account=pr_230_general

CONFIG_PATH=/scratch/rn2214/source-separation/SCNet/conf/config_small.yaml
SAVE_PATH=/scratch/rn2214/source-separation/SCNet/ckpt/binaural_scnet_small/

../sing << EOF

conda activate scnet

export SSL_CERT_FILE=/scratch/rn2214/cacert.pem

accelerate launch -m scnet.train --config_path $CONFIG_PATH \
                                 --save_path $SAVE_PATH

EOF
