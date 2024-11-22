import os

import jax

if jax.process_index()==0:
    os.system('ls test/JAX-SiT')