# Pointnet Custom Ops

This is a repackaging of the wonderful work [here](https://github.com/charlesq34/pointnet2) based on [tensorflow/custom-op](https://github.com/tensorflow/custom-op).

## Setup

After installing tensorflow (it will not be installed manually)

```bash
git clone https://github.com/jackd/pointnet_ops.git
cd pointnet_ops
make pip_pkg
pip install artifacts/pointnet_ops-<VERSION>.whl
```

## Usage

```python
from pointnet import sample
from pointnet import group
from pointnet import interpolate

...
```
