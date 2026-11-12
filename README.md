# Bi-SCNet

**Binaural Sparse Compression Network for Music Source Separation**

See: [Spacing Out: On the Reliability of Binaural Music Source Separation Metrics](https://github.com/richa-namballa/binaural-mss/)

**Contributions:**
* [Spatial Loss](https://github.com/richa-namballa/Bi-SCNet/blob/d75678e19f969bd4a1c981f794724d4aefb6ef86/scnet/loss.py#L142) based on ["Interaural Time Difference Loss for Binaural Target Sound Extraction"](https://ieeexplore.ieee.org/abstract/document/10694612) by Hernandez-Olivan et al. (2024)
* Weights and Biases Logging

This repository is an adaptation of the [SCNet: Sparse Compression Network for Music Source Separation](https://github.com/starrytong/SCNet) repository by Tong, et al. (2024):

```bibtex
@inproceedings{tong2024scnet,
  title={SCNet: Sparse Compression Network for Music Source Separation},
  author={Tong, Weinan and Zhu, Jiaxu and Chen, Jun and Kang, Shiyin and
  Jiang, Tao and Li, Yang and Wu, Zhiyong and Meng, Helen},
  booktitle={2024 IEEE International Conference on Acoustics,
  Speech and Signal Processing (ICASSP)},
  pages={1276--1280},
  year={2024},
  organization={IEEE}
}
```

---

## Set-Up

First, you need to install the requirements.

```bash
pip install -r requirements.txt
```

We use the accelerate package from Hugging Face for multi-GPU training.

```bash
accelerate config
```

You need to modify the dataset path in the /conf/config.yaml. The dataset folder should contain the "train" and "valid" splits.

```bash
data:
  wav: /path/to/dataset
```

---

## Training

The training command is as follows. If you do not specify a path, the default path will be used.

```bash
accelerate launch -m scnet.train \
                  --config_path path/to/config.yaml \
                  --save_path path/to/save/checkpoint/
```

---

## Inference

**Binaural Model Checkpoint**

The binaural model (Bi-SCNet) checkpoints are available on Hugging Face.
Both model versions use the same configuration file in `conf/`.

[Download Binaural Model Checkpoints](https://huggingface.co/richa-namballa/Bi-SCNet) 

**Stereo Model Checkpoint**

The stereo model checkpoint was trained on the MUSDB dataset by the original authors of SCNet. You can download their checkpoint from their repository [here](https://github.com/starrytong/SCNet#inference).

**Running Inference**

```bash
python -m scnet.inference \
          --input_dir path/to/test/dir \
          --output_dir path/to/save/result/ \
          --checkpoint_path path/to/checkpoint.th \
          --config_path path/to/config.yaml
```
