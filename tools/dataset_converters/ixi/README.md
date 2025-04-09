# Preparing IXI registration Dataset

<!-- [DATASET] -->

```bibtex
@Article{chen2022tranmorph,
title = {TransMorph: Transformer for unsupervised medical image registration},
journal = {Medical Image Analysis},
volume = {82},
pages = {102615},
year = {2022},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2022.102615},
author = {Junyu Chen and Eric C. Frey and Yufan He and William P. Segars and Ye Li and Yong Du},
}
```

The IXI datasets preprocessed by the TransMorph author can be download from their [Google Drive (1.44G)](https://github.com/RenYang-home/NTIRE21_VEnh).

For more details, please refer to [homepage](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_ IXI.md).

The folder structure should look like:

```text
mmipt
├── mmipt
├── tools
├── configs
├── data
|   ├── IXI_data
|   |   ├── Train
|   |   |   ├── subject_0.pkl
|   |   |   ├── ...
|   |   |   └── subject_9.pkl
|   |   ├── Val
|   |   |   ├── subject_105.pkl
|   |   |   ├── ...
|   |   |   └── subject_92.pkl
|   |   ├── Test
|   |   |   ├── subject_110.pkl
|   |   |   ├── ...
|   |   |   └── subject_99.pkl
|   |   ├── atlas.pkl
|   |   ├── dataset_info.txt
|   |   ├── label_info.txt
|   |   └── license.txt
```
