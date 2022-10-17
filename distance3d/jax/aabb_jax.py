import numpy as np
from jax import grad, jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'viridis'
rcParams['axes.grid'] = False


@jit
def _loop_aabbs_overlap(aabbs1, aabbs2):
    i_len = len(aabbs1)
    j_len = len(aabbs2)
    result = jnp.empty((i_len, j_len), dtype=bool)

    for i in range(i_len):
        for j in range(j_len):
            r = _aabb_overlap(aabbs1[i], aabbs2[j])
            result = result.at[i, j].set(r)

    return result

@jit
def _aabb_overlap(aabb1, aabb2):
    temp = aabb1[:, 1]
    aabb1 = aabb1.at[:, 1].set(aabb2[:, 1])
    aabb2 = aabb2.at[:, 1].set(temp)

    a = jnp.append(aabb1, aabb2, axis=0)
    b = a[:, 0] <= a[:, 1]

    return b.all()


def create_test_aabbs():
    aabbs = jnp.array([[[0, 2], [0, 2], [0, 1]],
                      [[1, 3.0], [1, 3], [0, 1]],
                      [[3, 4.0], [0, 1], [0, 1]]], dtype=float)

    aabbs2 = jnp.array([[[1.5, 3.5], [0, 0.5], [0, 1]],
                       [[1.5, 2], [2.5, 4], [0, 1]],
                       [[3.5, 4.5], [0, 0.5], [0, 1]]], dtype=float)
    return aabbs, aabbs2


aabb1, aabb2 = create_test_aabbs()
print(_loop_aabbs_overlap(aabb1, aabb2))





