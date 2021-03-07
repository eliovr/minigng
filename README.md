# Mini GNG
A simple version of the Growing Neural Gas algorithm by [Firtzke (1995)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.648.1905&rep=rep1&type=pdf) -- with some updates based on a submitted paper. 

## Requirements
### Mingng
- Numpy
- Scipy

### Example
- Pandas
- Matplotlib
- Networkx
- Graphviz

## Example use
```python
import pandas as pd
import numpy as np
from minigng import MiniGNG

df = pd.read_csv("/path/to/iris.csv", header=1)
training = df.drop(df.columns[-1], 1).to_numpy(dtype=np.float32)

gng = MiniGNG(max_units=40, n_epochs=30)
gng.fit(training)
gng.save_gml('iris.gml')
```

## Screenshots

[Iris](https://archive.ics.uci.edu/ml/datasets/Iris) dataset
```python
gng = MiniGNG(max_units=40, n_epochs=30)
```
![alt text](img/iris.png)

[MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
```python
gng = MiniGNG(max_units=150, n_epochs=25, max_edge_age=50, sample=.2, untangle=True, untangle_net_size=5)
```
![alt text](img/mnist.png)


[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.
```python
gng = MiniGNG(max_units=150, n_epochs=20, max_edge_age=30, sample=.3, untangle=True, untangle_net_size=0)
```
![alt text](img/fashion_mnist.png)