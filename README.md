# DilateMorph: A Dual-Stream Transformer with Multi-Dilation Cross-Attention for Medical Image Registration

 <img src="docs/imgs/dilatemorph.png" width = "800"  align=center />


## Installation

1. Follow the [official guide](https://pytorch.org/get-started/locally/) to install PyTorch.
2. Install MMIPT
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```


## Quick Start

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
</details>

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
</details>


## Directory structure

<details>
<summary>details</summary>

```bash
├── configs                                 Commonly used base config file.
├── mmipt
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── datasets.py                     Customize your dataset here
│   │   └── transforms.py                   Customize your data transform here
│   ├── engine
│   │   ├── __init__.py
│   │   ├── hooks.py                        Customize your hooks here
│   │   ├── optimizers.py                   Less commonly used. Customize your optimizer here
│   │   ├── optim_wrappers.py               Less commonly used. Customize your optimizer wrapper here
│   │   ├── optim_wrapper_constructors.py   Less commonly used. Customize your optimizer wrapper constructor here
│   │   └── schedulers.py                   Customize your lr/momentum scheduler here
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py                    Less commonly used. Customize your evaluator here
│   │   └── metrics.py                      Customize your metric here.
│   ├── models
│   │   ├── __init__.py
│   │   ├── model.py                        Customize your model here.
│   │   ├── weight_init.py                  Less commonly used here. Customize your initializer here.
│   │   └── wrappers.py                     Less commonly used here. Customize your wrapper here.
│   ├── __init__.py
│   ├── registry.py
│   ├── version.py
|   ```
├── tools                                   General train/test script
```
</details>


## Citation

```bibtex
@Article{paper,
  title = {DilateMorph: A Dual-Stream Transformer with Multi-Dilation Cross-Attention for Medical Image Registration},
  author = {},
  journal = {},
  volume = {},
  pages = {},
  year = {},
}
```
