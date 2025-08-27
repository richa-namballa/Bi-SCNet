#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 64GB
#SBATCH --time 47:59:00
#SBATCH --gres gpu:2
#SBATCH --job-name=scnet_train
#SBATCH --output logs/job.%J.scnet.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rn2214@nyu.edu
#SBATCH --account=pr_230_general

CONFIG_PATH=/scratch/rn2214/source-separation/SCNet/conf/config.yaml
SAVE_PATH=/scratch/rn2214/source-separation/SCNet/ckpt/

../sing << EOF

conda activate scnet

export SSL_CERT_FILE=/scratch/rn2214/cacert.pem

export NUMBA_CACHE_DIR=/scratch/rn2214/source-separation/SCNet/tmp

accelerate launch -m scnet.train --config_path $CONFIG_PATH \
                                 --save_path $SAVE_PATH

EOF
