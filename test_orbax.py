import os
import time
from os import times

import jax
import numpy as np
import tqdm
from PIL import Image
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from flax.training.common_utils import shard_prng_key
from jax.experimental import multihost_utils
from orbax.checkpoint.utils import fully_replicated_host_local_array_to_global_array

jax.distributed.initialize()


time.sleep(jax.process_index())

multihost_utils.sync_global_devices('sync device')

print(jax.process_index())

"""
rng=jax.random.PRNGKey(0)+jax.process_index()

rng=shard_prng_key(rng)



@jax.pmap
def go(rng):
    return    rng


rng=go(rng)


rng=fully_replicated_host_local_array_to_global_array(rng)

if jax.process_index()==0:
    print(rng,rng.devices())


while True:
    pass


checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

ckpt = {
    'rng': rng,
}
# orbax_checkpointer = ocp.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
checkpointer.save('gs://roger-center-2b/test_orbax', ckpt, save_args=save_args, force=True)



ckpt=checkpointer.restore('gs://roger-center-2b/test_orbax',item=ckpt)
restore_rng=ckpt['rng']

print(f'{restore_rng=}   {rng=}')


if jax.process_index()==0:
    print('\n'*5)

    print(f'{restore_rng=}   {rng=}')

    print('\n'*5)

"""
