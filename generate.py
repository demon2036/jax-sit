import argparse
import functools
import glob
import math
import os
import threading
from pathlib import Path

import PIL
import einops
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import webdataset as wds
from diffusers import FlaxAutoencoderKL
from flax.jax_utils import replicate
from flax.training import orbax_utils
from flax.training.common_utils import shard_prng_key
from orbax.checkpoint.utils import fully_replicated_host_local_array_to_global_array
from webdataset import TarWriter

from models_jax.convert_torch_to_jax import convert_torch_to_flax_sit
from models_jax.sit import SiT
from samplers_jax import euler_maruyama_sampler4
from utils import download_model
import orbax.checkpoint as ocp

lock = threading.Lock()


def send_file(keep_files=2, remote_path='shard_path2', rng=None, sample_rng=None, label=None, checkpointer=None):
    with lock:
        files = glob.glob('shard_path/*.tar')
        files.sort(key=lambda x: os.path.getctime(x), )

        if len(files) == 0:
            raise NotImplemented()
        elif len(files) <= keep_files:
            pass
        else:

            if keep_files == 0:
                files = files
            else:
                files = files[:-keep_files]
            # print(files)
            dst = remote_path
            if 'gs' not in remote_path:
                dst = os.getcwd() + '/' + dst
                os.makedirs(dst, exist_ok=True)

            for file in files:
                base_name = os.path.basename(file)

                if jax.process_index() == 0:
                    print(base_name, files)

                def send_data_thread(src_file, dst_file):
                    with wds.gopen(src_file, "rb") as fp_local:
                        data_to_write = fp_local.read()

                    with wds.gopen(f'{dst_file}', "wb") as fp:
                        fp.write(data_to_write)
                        # fp.flush()

                    os.remove(src_file)

                # send_data_thread(file, f'{dst}/{base_name}')
                threading.Thread(target=send_data_thread, args=(file, f'{dst}/{base_name}')).start()

            if rng is not None:
                rng = fully_replicated_host_local_array_to_global_array(rng)
                ckpt = {
                    'rng': rng,
                    # 'sample_rng': sample_rng,
                    'label': label - keep_files
                }
                # orbax_checkpointer = ocp.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                checkpointer.save(f'{dst}/resume.json', ckpt, save_args=save_args, force=True)

class CustomShardWriter(wds.ShardWriter):
    """
    CustomShardWriter to make it suitable for increase shard counter step by  jax.process_count()
    """
    def __init__(self, progress_count, *args, **kwargs):
        self.progress_count = progress_count
        super().__init__(*args, **kwargs)

    def next_stream(self):
        # print('her is next stream')
        """Close the current stream and move to the next."""
        self.finish()
        self.fname = self.pattern % self.shard
        if self.verbose:
            print(
                "# writing",
                self.fname,
                self.count,
                "%.1f GB" % (self.size / 1e9),
                self.total,
            )
        self.shard += self.progress_count
        self.tarstream = TarWriter(self.fname, **self.kw)
        self.count = 0
        self.size = 0


def test_convert(args):
    batch_per_worker = jax.local_device_count() * args.batch_per_core
    batch_per_all = args.batch_per_core * jax.device_count()
    iteration = math.ceil(args.num_samples / args.batch_per_core / jax.device_count())
    b, c, h, w = batch_per_worker, 4, 32, 32

    print(f'{threading.active_count()=}')
    # jax.distributed.initialize()

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

    rng = jax.random.PRNGKey(args.global_seed) + jax.process_index()
    rng = shard_prng_key(rng)
    total = 0

    sampling_kwargs = dict(
        model=model_jax,
        # latents=z,
        # y=y,
        num_steps=250,
    )

    params_sit_jax = replicate(params_sit_jax)
    vae_params = replicate(vae_params)
    sample_fn = functools.partial(euler_maruyama_sampler4, **sampling_kwargs)

    shard_dir_path = Path('shard_path')
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / 'shards-%05d.tar')
    print(shard_filename)

    @jax.pmap
    def go(model_params, vae_params, rng):
        rng, new_rng, rng_label, rng_sample = jax.random.split(rng, 4)
        z = jax.random.normal(rng, (args.batch_per_core, c, h, w))

        # y = jax.random.randint(rng_label, (args.batch_per_core,), 0, 999, jnp.int32)
        y = jnp.full((b,), 2, jnp.int32)

        samples_jax = sample_fn(model_params=model_params, latents=z, y=y, rng=rng_sample)
        latent = samples_jax / 0.18215
        img = vae_flax.apply({'params': vae_params}, latent, method=vae_flax.decode).sample

        img = (img + 1) / 2.

        img = jnp.clip(img * 255, 0, 255)

        return img, y, new_rng

    counter = 0
    lock = threading.Lock()

    def thread_write(images, class_labels, sink):
        images = np.array(images).astype(np.uint8)
        class_labels = np.array(class_labels)

        shard_idx = sink.shard

        with lock:
            nonlocal counter

            for img, cls_label in zip(images, class_labels):
                sink.write({
                    "__key__": "%010d" % counter,
                    "jpg": PIL.Image.fromarray(img),
                    "cls": int(cls_label),
                })
                counter += 1

                # if shard_idx!=sink.shard:
                #     print(f'here is stop !!! {shard_idx=} {sink.shard=}')
                #
                #     while True:
                #         pass

            if jax.process_index() == 0:
                print(counter, images.shape)

            # if send_file:
            #     sink.shard = jax.process_index() + (label + 1) * jax.process_count()
            # sink.next_stream()
            # thread_send()

    data_per_shard = args.data_per_shard
    # per_process_generate_data = b * jax.local_device_count()
    # assert data_per_shard % per_process_generate_data == 0
    # iter_per_shard = data_per_shard // per_process_generate_data

    sink = CustomShardWriter(
        pattern=shard_filename,
        maxcount=data_per_shard,
        maxsize=3e10,
        start_shard=jax.process_index(),
        verbose=jax.process_index() == 0,
        progress_count=jax.process_count()
        # maxsize=shard_size,
    )

    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())



    for i in tqdm.tqdm(range(iteration)):
        samples_jax, labels, rng = go(params_sit_jax, vae_params, rng)

        samples_jax = einops.rearrange(samples_jax, 'n b c h w -> (n b) h w c')
        labels = einops.rearrange(labels, 'n b  -> (n b) ')
        # print(samples_jax.shape,labels.shape)

        threading.Thread(target=thread_write,
                         args=(
                             samples_jax, labels, sink,)).start()

        send_file(3, args.output_dir, rng, sample_rng=None, label=i, checkpointer=checkpointer)

    """
    
    start_label=0
    if args.resume:
        dst = args.output_dir + '/' + 'resume.json'
        if 'gs' not in dst:
            dst = os.getcwd() + '/' + dst
        ckpt = {
            'rng': rng,
            'sample_rng': sample_rng,
            'label': 1
        }
        ckpt = checkpointer.restore(dst, item=ckpt)
        rng = ckpt['rng']
        sample_rng = ckpt['sample_rng']
        start_label=ckpt['label']
        # print(ckpt)
    """

    """

abel in tqdm.trange()
    for label in range(start_label, args.per_process_shards):
        print(label)

        for i in tqdm.tqdm(range(iter_per_shard), disable=not jax.process_index() == 0):
            rng, sample_rng, images, class_labels = test_sharding_jit(rng, sample_rng, converted_jax_params, vae_params,
                                                                      label)

            local_images = collect_process_data(images)
            local_class_labels = collect_process_data(class_labels)
            threading.Thread(target=thread_write,
                             args=(
                                 local_images, local_class_labels, sink, label,
                                 True if i == iter_per_shard - 1 else False)).start()
        send_file(3, args.output_dir, rng, sample_rng, label, checkpointer)

    while threading.active_count() > 2:
        print(f'{threading.active_count()=}')
        time.sleep(1)
    sink.close()
    print('now send file')
    send_file(0, args.output_dir, rng, sample_rng, label, checkpointer)
    while threading.active_count() > 2:
        print(f'{threading.active_count()=}')
        time.sleep(1)
    """


if __name__ == "__main__":
    jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    # parser.add_argument("--output-dir", default="shard_path2")
    # parser.add_argument("--output-dir", default="gs://shadow-center-2b/imagenet-generated-100steps-cfg1.75")

    parser.add_argument("--output-dir", default="gs://roger-center-2b/imagenet-generated-sit-250steps")
    # parser.add_argument("--seed", type=int, default=7)
    # parser.add_argument("--sample-seed", type=int, default=24)
    # parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--data-per-shard", type=int, default=1024)  #2048
    # parser.add_argument("--per-process-shards", type=int, default=400)
    # parser.add_argument("--per-device-batch", type=int, default=128)  #128
    # parser.add_argument("--resume",  action="store_true", default=False)

    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--batch-per-core", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=1000000)

    test_convert(parser.parse_args())
