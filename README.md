# Mini GNG
A simple version of the Growing Neural Gas algorithm by [Firtzke (1995)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.648.1905&rep=rep1&type=pdf).

## Example use
```python
import pandas as pd
from minigng import MiniGNG

df = pd.read_csv("/path/to/iris.csv", header=1)
training = df.drop(df.columns[-1], 1).to_numpy()

gng = MiniGNG(max_units=40, n_epochs=30)
gng.fit(training)
gng.save_gml('iris.gml')
```
