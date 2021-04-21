# One-stage-Mask-based-Scene-Text-Eraser
This is the source code of [MTRNet++: One-stage Mask-based Scene Text Eraser](https://arxiv.org/abs/1912.07183)

Note some parts of the codes are taken from [edge-connect](https://github.com/knazeri/edge-connect).

If you need further help, please raise an issue.

## Prerequisites
- Python 3
- PyTorch 1.0
- NVIDIA GPU + CUDA cuDNN

### Other Networks
You can find network architectures for [MTRNet](https://arxiv.org/abs/1903.04092) and [ENSNet](https://arxiv.org/abs/1812.00723) under Related Networks. If you want to test these codes, you need to replace them with MTRNet++ architecture under `src/networks`.

### Citation

```
@article{tursun2020mtrnet++,
  title={MTRNet++: One-stage mask-based scene text eraser},
  author={Tursun, Osman and Denman, Simon and Zeng, Rui and Sivapalan, Sabesan and Sridharan, Sridha and Fookes, Clinton},
  journal={Computer Vision and Image Understanding},
  volume={201},
  pages={103066},
  year={2020},
  publisher={Academic Press}
}
```

```
@inproceedings{tursun2019mtrnet,
  title={Mtrnet: A generic scene text eraser},
  author={Tursun, Osman and Zeng, Rui and Denman, Simon and Sivapalan, Sabesan and Sridharan, Sridha and Fookes, Clinton},
  booktitle={2019 International Conference on Document Analysis and Recognition (ICDAR)},
  pages={39--44},
  year={2019},
  organization={IEEE}
}
```

