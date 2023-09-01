# QuanNet

## Installation

1. Install [APBS 3.4.1](https://apbs.readthedocs.io/en/latest/getting/index.html#installing-from-pre-compiled-binaries)
from pre-compiled binaries
2. Export environment variables for:
   * Python interpreter with installed dependencies
   * APBS binary file
   ```commandline
   export PYTHON=/path/to/interpreter
   export APBS_PATH=/path/APBS-3.4.1.Linux/bin/apbs
   ```

4. Install quannet Python package:

```commandline
pip install -e .
```

## Usage:

### Train:
```commandline
quannet mode=train data=quannet_test
```

### Predict:
```commandline
quannet mode=predict model=models/quannet.pt structures=datasets/quannet_test/structures
```
