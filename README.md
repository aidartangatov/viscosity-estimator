# QuanNet

See below installation and usage example

<details open>
<summary>Install</summary>



Build docker image and pip install the quannet package:


```bash
docker build -t quannet .
pip install .
```

</details>

<details open>
<summary>Usage</summary>

### CLI

QuanNet may be used directly in the Command Line Interface (CLI) with a `quannet` command:

#### Predict

```bash
quannet predict model=models/quannet.pt structures=datasets/quannet_test/structures
```

#### Train

```bash
quannet train dataset=datasets/quannet_test
```



### Python

QuanNet may also be used directly in a Python environment, and accepts the same as in the CLI example above:

```python
from quannet import QuanNet

# Load a model
model = QuanNet("models/quannet.pt")

# Use the model
model.predict(structures="datasets/quannet_test/structures")
model.train(dataset="datasets/quannet_test", epochs=3)

```
