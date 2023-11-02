import jax.numpy as jnp
import numpy as np

N = 7000
jnp.linalg.svd(np.random.random(size=(N, N)))

print('yay')