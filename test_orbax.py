import os

import jax
import numpy as np
import tqdm
from PIL import Image
import orbax.checkpoint as ocp

jax.distributed.initialize()

rng=jax.random.PRNGKey(0)+jax.process_index()

print(rng)

checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())



checkpointer.save('gs://roger-center-2b/test_orbax',{
    'rng':rng
},force=True)

restore_rng=checkpointer.restore('gs://roger-center-2b/test_orbax')
print(f'{restore_rng=}   {rng=}')

