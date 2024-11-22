import os

import jax
import numpy as np
import tqdm
from PIL import Image
import orbax.checkpoint as ocp
from flax.training.common_utils import shard_prng_key
from orbax.checkpoint.utils import fully_replicated_host_local_array_to_global_array

jax.distributed.initialize()

rng=jax.random.PRNGKey(0)+jax.process_index()

rng=shard_prng_key(rng)
rng=fully_replicated_host_local_array_to_global_array(rng)

print(rng)

checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())



checkpointer.save('gs://roger-center-2b/test_orbax',{
    'rng':rng
},force=True)

restore_rng=checkpointer.restore('gs://roger-center-2b/test_orbax')
print(f'{restore_rng=}   {rng=}')

