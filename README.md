# One-stage-Mask-based-Scene-Text-Eraser
This is the source code of [MTRNet++: One-stage Mask-based Scene Text Eraser](https://arxiv.org/abs/1912.07183)

Note some parts of the codes are taken from [edge-connect](https://github.com/knazeri/edge-connect).

In this work, we have used the Oxford Synthetic Text dataset and SCUT text removal dataset. To create necessary files for the training and testing, please check related issues, scripts, and files. You can use ```scripts/create_Oxford_dataset.py``` to generate files for Oxford Synthetic dataset. For the SCUT dataset, you can use the files in the ```data``` folder. 

If you need further help, please raise an issue or send email to me. 

## Prerequisites
- Python 3
- PyTorch 1.0
- NVIDIA GPU + CUDA cuDNN

### Other Networks
You can find network architectures for [MTRNet](https://arxiv.org/abs/1903.04092) and [ENSNet](https://arxiv.org/abs/1812.00723) under Related Networks. If you want to test these codes, you need to replace them with MTRNet++ architecture under `src/networks`. Note that the original MTRNet is implemented with Tensorflow and the backbone is U-Net. However, here, the MTRNet is using the same backbone of the MTRNet++ which is a lighter backbone compared to MTRNet.

### Citation

If you find our code useful to your research, please cite our papers [MTRNet](https://arxiv.org/abs/1903.04092) and [MTRNet++](https://arxiv.org/abs/1912.07183): 

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

