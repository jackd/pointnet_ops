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

You will need `$CUDA_HOME` defined (generally `/usr/local/cuda`). You may also need to change `g++` version at the top of the make file. See [build configurations](https://www.tensorflow.org/install/source#tested_build_configurations) if you install tensorflow from `pip`.

## Usage

```python
from pointnet_ops import sample
from pointnet_ops import group
from pointnet_ops import interpolate

...
```

## Alternative Implementations

Note the `pointnet_ops.group.group_point` and ``pointnet_ops.interpolate.three_interpolate` can be implemented significantly more easily using base tensorflow operations. See [examples](examples).
