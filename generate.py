import argparse
import functools
import math
import os

import einops
import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm
from PIL import Image
from diffusers import FlaxAutoencoderKL
from flax.jax_utils import replicate
from flax.training.common_utils import shard, shard_prng_key

from models_jax.convert_torch_to_jax import convert_torch_to_flax_sit
from models_jax.sit import SiT
from samplers_jax import euler_maruyama_sampler2, euler_maruyama_sampler3, jax_to_torch
from utils import download_model
from jax.experimental.multihost_utils import process_allgather








def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm.tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path





def generate(args):






    batch_per_worker=jax.local_device_count()*args.batch_per_core
    iteration=math.ceil(args.num_samples/args.batch_per_core/jax.device_count())

    b, c, h, w = batch_per_worker, 4, 32, 32


    # print(args.num_samples,args.batch_per_core/jax.process_count(),args.num_samples/args.batch_per_core/jax.process_count())
    #
    # while True:
    #     pass





    sample_folder_dir = f"test/JAX-SiT"
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    vae_flax, vae_params = FlaxAutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema", local_files_only=False,
                                                             local_dir='vae',
                                                             cache_dir='vae_flax', from_pt=True)

    vae_params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x)), vae_params)

    model_kwargs = {
        'input_size': 32,
        'patch_size': 2,
        'hidden_size': 1152,
        'depth': 28,
        'num_heads': 16,
        'class_dropout_prob': 0.1,
        'decoder_hidden_size': 1152
        # 'norm_layer':None
    }

    model_jax = SiT(**model_kwargs)

    state_dict = download_model('last.pt')

    params_torch = {k: v.cpu().numpy() for k, v in state_dict.items()}
    params_torch = flax.traverse_util.unflatten_dict(params_torch, sep=".")

    params_sit_jax = convert_torch_to_flax_sit(params_torch)

    # sampler=EluerMaruyamaSampler2()

    rng = jax.random.PRNGKey(args.global_seed)+jax.process_index()
    rng=shard_prng_key(rng)
    total=0



    sampling_kwargs = dict(
        model=model_jax,
        # latents=z,
        # y=y,
        num_steps=250,
    )

    params_sit_jax = replicate(params_sit_jax)
    sample_fn = functools.partial(euler_maruyama_sampler3, **sampling_kwargs)
    for _ in tqdm.tqdm(range(iteration)):

        @jax.pmap
        def go(model_params, rng):
            rng,new_rng,rng_label=jax.random.split(rng)
            z = jax.random.normal(rng, (args.batch_per_core, c, h, w))

            y = jax.random.normal(rng_label,(args.batch_per_core,),  jnp.int32)
            # y = jnp.full((b,), 2, jnp.int32)

            # samples_jax = sampler.apply({'params': {'model': params_sit_jax}})
            samples_jax = sample_fn(model_params=model_params, latents=z, y=y)
            latent = samples_jax / 0.18215
            img=vae_flax.apply({'params': vae_params}, latent, method=vae_flax.decode).sample
            return img,new_rng

        samples_jax,rng = go(params_sit_jax, rng)
        samples_jax=einops.rearrange(samples_jax,'n b c h w-> (n b) c h w')

        samples_jax = jax_to_torch(samples_jax)

        samples = samples_jax

        samples = (samples + 1) / 2.
        samples = torch.clamp(
            255. * samples, 0, 255
        ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        print(samples.shape)
        samples=process_allgather(samples)
        samples =einops.rearrange(samples,'n b c h w-> (n b) c h w')
        print(samples.shape)

        # Save samples to disk as individual .png files

        if jax.process_index()==0:
            for i, sample in enumerate(samples):
                index = i+total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total+=b
    if jax.process_index() == 0:
        create_npz_from_sample_folder(sample_folder_dir,total)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--batch-per-core", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=50000)
    args = parser.parse_args()

    # jax.distributed.initialize()

    generate(args)