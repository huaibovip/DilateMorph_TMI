# TransMorph

> [TransMorph: Transformer for unsupervised medical image registration](https://www.sciencedirect.com/science/Article/abs/pii/S1361841522002432)


## Abstract

<!-- [ABSTRACT] -->

In the last decade, convolutional neural networks (ConvNets) have been a major focus of research in medical image analysis. However, the performances of ConvNets may be limited by a lack of explicit consideration of the long-range spatial relationships in an image. Recently, Vision Transformer architectures have been proposed to address the shortcomings of ConvNets and have produced state-of-the-art performances in many medical imaging applications. Transformers may be a strong candidate for image registration because their substantially larger receptive field enables a more precise comprehension of the spatial correspondence between moving and fixed images. Here, we present TransMorph, a hybrid Transformer-ConvNet model for volumetric medical image registration. This paper also presents diffeomorphic and Bayesian variants of TransMorph: the diffeomorphic variants ensure the topology-preserving deformations, and the Bayesian variant produces a well-calibrated registration uncertainty estimate. We extensively validated the proposed models using 3D medical images from three applications: inter-patient and atlas-to-patient brain MRI registration and phantom-to-CT registration. The proposed models are evaluated in comparison to a variety of existing registration methods and Transformer architectures. Qualitative and quantitative results demonstrate that the proposed Transformer-based model leads to a substantial performance improvement over the baseline methods, confirming the effectiveness of Transformers for medical image registration.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/user-attachments/assets/fbb5b187-5d8c-48b4-b429-67c73dc6ecc5" width="70%" target="_blank"/>
<img src="https://github.com/user-attachments/assets/25070aa2-019f-4438-9dc1-d8c251cb9045" width="70%" target="_blank"/>
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
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/registration/transmorph/transmorph_ixi_atlas-to-scan_160x192x224.py

# single-gpu train
python tools/train.py configs/registration/transmorph/transmorph_ixi_atlas-to-scan_160x192x224.py

# multi-gpu train
./tools/dist_train.sh configs/registration/transmorph/transmorph_ixi_atlas-to-scan_160x192x224.py 4
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMipt).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/registration/transmorph/transmorph_ixi_atlas-to-scan_160x192x224.py path/to/checkpoint.pth

# single-gpu test
python tools/test.py configs/registration/transmorph/transmorph_ixi_atlas-to-scan_160x192x224.py path/to/checkpoint.pth

# multi-gpu test
./tools/dist_test.sh configs/registration/transmorph/transmorph_ixi_atlas-to-scan_160x192x224.py path/to/checkpoint.pth 4
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMipt).

</details>


## Citation

```bibtex
@Article{chen2022transmorph,
  title = {TransMorph: Transformer for unsupervised medical image registration},
  author = {Junyu Chen and Eric C. Frey and Yufan He and William P. Segars and Ye Li and Yong Du},
  journal = {Medical Image Analysis},
  volume = {82},
  pages = {102615},
  year = {2022},
  publisher={Elsevier},
  doi = {10.1016/j.media.2022.102615}
}
```
