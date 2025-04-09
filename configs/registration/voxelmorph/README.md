# VoxelMorph

> [VoxelMorph: A Learning Framework for Deformable Medical Image Registration (TMI)](https://ieeexplore.ieee.org/abstract/document/8633930)  
> [An Unsupervised Learning Model for Deformable Medical Image Registration (CVPR)](https://openaccess.thecvf.com/content_cvpr_2018/html/Balakrishnan_An_Unsupervised_Learning_CVPR_2018_paper.html)  
> [Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration (MICCAI)](https://link.springer.com/chapter/10.1007/978-3-030-00928-1_82)

## Abstract

<!-- [ABSTRACT] -->

We present VoxelMorph, a fast learning-based framework for deformable, pairwise medical image registration. Traditional registration methods optimize an objective function for each pair of images, which can be time-consuming for large datasets or rich deformation models. In contrast to this approach and building on recent learning-based methods, we formulate registration as a function that maps an input image pair to a deformation field that aligns these images. We parameterize the function via a convolutional neural network and optimize the parameters of the neural network on a set of images. Given a new pair of scans, VoxelMorph rapidly computes a deformation field by directly evaluating the function. In this paper, we explore two different training strategies. In the first (unsupervised) setting, we train the model to maximize standard image matching objective functions that are based on the image intensities. In the second setting, we leverage auxiliary segmentations available in the training data. We demonstrate that the unsupervised model's accuracy is comparable to the state-of-the-art methods while operating orders of magnitude faster. We also show that VoxelMorph trained with auxiliary data improves registration accuracy at test time and evaluate the effect of training set size on registration. Our method promises to speed up medical image analysis and processing pipelines while facilitating novel directions in learning-based registration and its applications. Our code is freely available at https://github.com/voxelmorph/voxelmorph.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/user-attachments/assets/f987004a-4a9a-4207-8401-d21c5af4d85d" width="70%" target="_blank"/>

Fig 1: VoxelMorph Overview
</div>

<!-- [ABSTRACT] -->

Traditional deformable registration techniques achieve impressive results and offer a rigorous theoretical treatment, but are computationally intensive since they solve an optimization problem for each image pair. Recently, learning-based methods have facilitated fast registration by learning spatial deformation functions. However, these approaches use restricted deformation models, require supervised labels, or do not guarantee a diffeomorphic (topology-preserving) registration. Furthermore, learning-based registration tools have not been derived from a probabilistic framework that can offer uncertainty estimates. In this paper, we present a probabilistic generative model and derive an unsupervised learning-based inference algorithm that makes use of recent developments in convolutional neural networks (CNNs). We demonstrate our method on a 3D brain registration task, and provide an empirical analysis of the algorithm. Our approach results in state of the art accuracy and very fast runtimes, while providing diffeomorphic guarantees and uncertainty estimates. Our implementation is available online at http://voxelmorph.csail.mit.edu.

<!-- [IMAGE] -->
<div align=center>
<img src="https://media.springernature.com/full/springer-static/image/chp%3A10.1007%2F978-3-030-00928-1_82/MediaObjects/473972_1_En_82_Fig1_HTML.png?as=webp" width="70%" target="_blank"/>

Fig 2: Diffeomorphic VoxelMorph Overview
</div>


## Results and models

### IXI

### LPBA40

### OASIS


## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/registration/voxelmorph/voxelmorph1_ixi_atlas-to-scan_160x192x224.py

# single-gpu train
python tools/train.py configs/registration/voxelmorph/voxelmorph1_ixi_atlas-to-scan_160x192x224.py

# multi-gpu train
./tools/dist_train.sh configs/registration/voxelmorph/voxelmorph1_ixi_atlas-to-scan_160x192x224.py 4
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMipt).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/registration/voxelmorph/voxelmorph1_ixi_atlas-to-scan_160x192x224.py path/to/checkpoint.pth

# single-gpu test
python tools/test.py configs/registration/voxelmorph/voxelmorph1_ixi_atlas-to-scan_160x192x224.py path/to/checkpoint.pth

# multi-gpu test
./tools/dist_test.sh configs/registration/voxelmorph/voxelmorph1_ixi_atlas-to-scan_160x192x224.py path/to/checkpoint.pth 4
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMipt).

</details>


## Citation

```bibtex
@Article{balakrishnan2019voxelmorph,
  title={VoxelMorph: A Learning Framework for Deformable Medical Image Registration}, 
  author={Balakrishnan, Guha and Zhao, Amy and Sabuncu, Mert R. and Guttag, John and Dalca, Adrian V.},
  journal={IEEE Transactions on Medical Imaging}, 
  volume={38},
  pages={1788-1800},
  number={8},
  year={2019},
  publisher={IEEE},
  doi={10.1109/TMI.2019.2897538}
}
@InProceedings{balakrishnan2018cvpr,
  title={An Unsupervised Learning Model for Deformable Medical Image Registration},
  author={Balakrishnan, Guha and Zhao, Amy and Sabuncu, Mert R. and Guttag, John and Dalca, Adrian V.},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={9252-9260},
  month={June},
  year={2018},
  doi={10.1109/CVPR.2018.00964}
}
@InProceedings{dalca2018miccai,
  title={Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration},
  author={Dalca, Adrian V. and Balakrishnan, Guha and Guttag, John and Sabuncu, Mert R.},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2018},
  publisher={Springer},
  address={Cham},
  pages={729-738},
  doi={10.1007/978-3-030-00928-1_82}
}
```
