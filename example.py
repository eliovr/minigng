import pandas as pd
import numpy as np
from minigng import MiniGNG

df = pd.read_csv("/path/to/iris.csv", header=1)
training = df.drop(df.columns[-1], 1).to_numpy(dtype=np.float32)

gng = MiniGNG(max_units=40, n_epochs=30)
gng.fit(training)
gng.save_gml('iris.gml')
