try: # sanity check
  import torch
  x = torch.rand(5, 3)
  print(x)
except ReferenceError:
  exit()

import torch
import numpy as np

# From raw data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print("raw:",{x_data, x_data.shape, x_data.dtype})

# From np ndarray
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print("array is tensor: ", isinstance(x_np, torch.Tensor))
