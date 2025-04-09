# 准备 IXI registration 数据集

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

由TransMorph作者预处理的IXI数据集可以从其[Google Drive (1.44G)](https://drive.google.com/uc?export=download&id=1-VQewCVNj5eTtc3eQGhTM2yXBQmgm8Ol)下载。

更多详细内容参考其[主页](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md)。

文件目录结构应如下所示：

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
