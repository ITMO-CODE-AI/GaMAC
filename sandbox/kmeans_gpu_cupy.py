import numpy as np
import cupy as cp

x_cpu = np.array([1, 2, 3])
x_gpu = cp.array([1, 2, 3])

l2_cpu = np.linalg.norm(x_cpu)
l2_gpu = cp.linalg.norm(x_gpu)

x = cp.arange(10, dtype=np.float32).reshape(2, 5)
y = cp.arange(5, dtype=np.float32)

squared_diff = cp.ElementwiseKernel(
   'float32 x, float32 y',
   'float32 z',
   'z = (x - y) * (x - y)',
   'squared_diff')

squared_diff(x, y)

print("Using Numpy: ", l2_cpu)
print("\nUsing Cupy: ", l2_gpu)