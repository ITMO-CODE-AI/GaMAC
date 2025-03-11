from __future__ import division
from numba import cuda
import numpy
import math
import pprint

DEVICE = {'name': cuda.get_current_device().name,
          'compute_capability': cuda.get_current_device().compute_capability,
          'id': cuda.get_current_device().id,
          }

pprint.pprint(DEVICE)


@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2


def main():
    print('///'*20)
    data = numpy.ones(256)
    threadsperblock = 256
    blockspergrid = math.ceil(data.shape[0] / threadsperblock)
    my_kernel[blockspergrid, threadsperblock](data)
    print(data)
    print('///'*20)

if __name__ == '__main__':
    main()