import numpy as np

from pathologybot_py.model import ImpalaModel, ModelSize

a = ImpalaModel(ModelSize.Smol)
inp = np.random.rand(1, 40, 40, 1)
y_pred = np.ones((1, 5))
a._model.fit(inp, y_pred)
