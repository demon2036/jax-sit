import einops
import flax
import torch

from models_jax.convert_torch_to_jax import *
from models_jax.sit import *
from models.sit import build_mlp,modulate as modulateTorch,TimestepEmbedder as TimestepEmbedderTorch,LabelEmbedder as LabelEmbedderTorch,SiTBlock as SiTBlockTorch,SiT as SiTTorch
from timm.layers import PatchEmbed as PatchEmbedTorch

from utils import download_model


def test_print(p, x):
    print(p)


def test_print2(p, x):
    print(p,x.shape)



def test_convert_mlp():
    shape = (1, 30, )
    x_torch = torch.rand(shape)
    x_jax = jnp.array(x_torch.detach().numpy())
    # x_jax = einops.rearrange(x_jax, 'b c h w -> b h w c')

    model_kwargs = {
        'hidden_size': 30,
        'projector_dim': 1000,
        'z_dim':100
        # 'norm_layer':None

    }

    rngs = jax.random.PRNGKey(1)

    model_torch = build_mlp(**model_kwargs,)
    params_torch = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params_torch = flax.traverse_util.unflatten_dict(params_torch, sep=".")

    model_jax = MLP(**model_kwargs)
    params_jax = model_jax.init(rngs, x_jax)['params']
    print(model_torch.state_dict().keys())

    jax.tree_util.tree_map_with_path(test_print2,params_jax)
    params_jax = convert_torch_to_flax_mlp(params_torch)
    jax.tree_util.tree_map_with_path(test_print2,params_jax)

    out_jax = model_jax.apply({'params': params_jax}, x_jax)
    out_jax = np.array(out_jax)
    out_torch = model_torch(x_torch).detach().numpy()
    # out_torch = einops.rearrange(out_torch, 'b c h w -> b h w c')

    print(out_torch.shape)
    print(out_jax.shape)
    print(np.max(np.abs(out_jax - out_torch)))
    np.testing.assert_almost_equal(out_jax, out_torch, decimal=6)



def test_convert_temb():
    shape = (1, )
    x_torch = torch.rand(shape)
    x_jax = jnp.array(x_torch.detach().numpy())
    # x_jax = einops.rearrange(x_jax, 'b c h w -> b h w c')

    model_kwargs = {
        'hidden_size': 300,
        # 'projector_dim': 10,
        # 'z_dim':100
        # 'norm_layer':None

    }

    rngs = jax.random.PRNGKey(1)

    model_torch = TimestepEmbedderTorch(**model_kwargs,)


    model_jax = TimestepEmbedder(**model_kwargs)
    params_jax = model_jax.init(rngs, x_jax)['params']


    params_torch = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params_torch = flax.traverse_util.unflatten_dict(params_torch, sep=".")
    print(model_torch.state_dict().keys())

    def test_print(p, x):
        print(p)

    jax.tree_util.tree_map_with_path(test_print,params_jax)
    params_jax = convert_torch_to_flax_lemb(params_torch)

    out_jax = model_jax.apply({'params': params_jax}, x_jax)
    out_jax = np.array(out_jax)
    out_torch = model_torch(x_torch).detach().numpy()
    # out_torch = einops.rearrange(out_torch, 'b c h w -> b h w c')

    print(out_torch.shape)
    print(out_jax.shape)
    print(np.max(np.abs(out_jax - out_torch)))
    np.testing.assert_almost_equal(out_jax, out_torch, decimal=6)


def test_convert_lemb():
    shape = (1, )
    # x_torch = torch.rand(shape)
    x_torch = torch.ones(shape,dtype=torch.int64)
    x_jax = jnp.array(x_torch.detach().numpy())
    # x_jax = einops.rearrange(x_jax, 'b c h w -> b h w c')

    model_kwargs = {
        'num_classes': 1000,
        'hidden_size': 10,
        'dropout_prob':0.0
        # 'norm_layer':None

    }

    rngs = jax.random.PRNGKey(1)

    model_torch = LabelEmbedderTorch(**model_kwargs,)
    params_torch = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params_torch = flax.traverse_util.unflatten_dict(params_torch, sep=".")

    model_jax = LabelEmbedder(**model_kwargs)
    params_jax = model_jax.init(rngs, x_jax,True)['params']
    print(model_torch.state_dict().keys())



    jax.tree_util.tree_map_with_path(test_print,params_jax)
    params_jax = convert_torch_to_flax_lemb(params_torch)

    out_jax = model_jax.apply({'params': params_jax}, x_jax,True,rngs={'dropout':rngs})
    out_jax = np.array(out_jax)
    out_torch = model_torch(x_torch,True).detach().numpy()
    # out_torch = einops.rearrange(out_torch, 'b c h w -> b h w c')

    print(out_torch.shape)
    print(out_jax.shape)
    print(np.max(np.abs(out_jax - out_torch)))
    np.testing.assert_almost_equal(out_jax, out_torch, decimal=6)



def test_convert_patch_embed():
    shape = (1, 3, 32, 32)
    x_torch = torch.rand(shape)
    x_jax = jnp.array(x_torch.detach().numpy())
    x_jax = einops.rearrange(x_jax, 'b c h w -> b h w c')

    model_kwargs = {
        'img_size': 32,
        'patch_size': 2,
        'embed_dim':768
        # 'norm_layer':None

    }

    rngs = jax.random.PRNGKey(1)

    model_torch = PatchEmbedTorch(**model_kwargs,)
    params_torch = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params_torch = flax.traverse_util.unflatten_dict(params_torch, sep=".")

    # model_kwargs.pop('in_channels')
    model_jax = PatchEmbed(**model_kwargs)
    params_jax = model_jax.init(rngs, x_jax)['params']
    print(model_torch.state_dict().keys())

    def test_print(p, x):
        print(p)

    jax.tree_util.tree_map_with_path(test_print2,params_jax)
    params_jax = convert_torch_to_flax_patch_embed(params_torch)
    jax.tree_util.tree_map_with_path(test_print2,params_jax)

    out_jax = model_jax.apply({'params': params_jax}, x_jax)
    out_jax = np.array(out_jax)
    out_torch = model_torch(x_torch).detach().numpy()
    # out_torch = einops.rearrange(out_torch, 'b c h w -> b h w c')

    print(out_torch.shape)
    print(out_jax.shape)
    print(np.max(np.abs(out_jax - out_torch)))
    np.testing.assert_almost_equal(out_jax, out_torch, decimal=6)











def test_convert_sit_block():
    shape = (1, 128, 256)
    x_torch = torch.rand(shape)
    c_torch = torch.rand((1,256))

    x_jax = jnp.array(x_torch.detach().numpy())
    c_jax = jnp.array(c_torch.detach().numpy())
    # x_jax = einops.rearrange(x_jax, 'b c h w -> b h w c')

    model_kwargs = {
        'hidden_size': 256,
        'num_heads': 4,
    }

    block_kwargs={'qk_norm':False,'fused_attn':False}

    rngs = jax.random.PRNGKey(1)

    model_torch = SiTBlockTorch(**model_kwargs,**block_kwargs)
    params_torch = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params_torch = flax.traverse_util.unflatten_dict(params_torch, sep=".")

    # model_kwargs.pop('in_channels')
    model_jax = SiTBlock(**model_kwargs)
    params_jax = model_jax.init(rngs, x_jax,c_jax)['params']
    print(model_torch.state_dict().keys())

    def test_print(p, x):
        print(p)

    jax.tree_util.tree_map_with_path(test_print2,params_jax)
    params_jax = convert_torch_to_flax_sit_block(params_torch)
    jax.tree_util.tree_map_with_path(test_print2,params_jax)

    out_jax = model_jax.apply({'params': params_jax}, x_jax,c_jax)
    out_jax = np.array(out_jax)
    out_torch = model_torch(x_torch,c_torch).detach().numpy()
    # out_torch = einops.rearrange(out_torch, 'b c h w -> b h w c')

    print(out_torch.shape)
    print(out_jax.shape)
    print(np.max(np.abs(out_jax - out_torch)))
    np.testing.assert_almost_equal(out_jax, out_torch, decimal=6)









def test_convert_sit():
    shape = (2, 4, 32, 32)
    x_torch = torch.rand(shape)
    t_torch = torch.ones(2,)*0.5
    y_torch = torch.ones(2,dtype=torch.int64)*2

    x_jax = jnp.array(x_torch.detach().numpy())
    t_jax = jnp.array(t_torch.detach().numpy())
    y_jax = jnp.array(y_torch.detach().numpy())
    x_jax = einops.rearrange(x_jax, 'b c h w -> b h w c')

    model_kwargs = {
        'input_size': 32,
        'patch_size': 2,
        'hidden_size':1152,
        'depth':28,
        'num_heads':16,
        'class_dropout_prob':0.1,
        'decoder_hidden_size':1152
        # 'norm_layer':None
    }

    block_kwargs={'qk_norm':False,'fused_attn':False}

    rngs = jax.random.PRNGKey(1)

    model_torch = SiTTorch(**model_kwargs,**block_kwargs)




    state_dict = download_model('last.pt')
    model_torch.load_state_dict(state_dict)
    model_torch.eval()  # important!



    params_torch = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params_torch = flax.traverse_util.unflatten_dict(params_torch, sep=".")

    model_jax = SiT(**model_kwargs)
    params_jax = model_jax.init(rngs, x_jax,t_jax,y_jax)['params']
    print(model_torch.state_dict().keys())

    def test_print(p, x):
        print(p)

    jax.tree_util.tree_map_with_path(test_print,params_jax)
    params_jax = convert_torch_to_flax_sit(params_torch)

    out_jax = model_jax.apply({'params': params_jax}, x_jax,t_jax,y_jax,rngs={'dropout':rngs})
    out_jax = np.array(out_jax)
    out_torch = model_torch(x_torch,t_torch,y_torch).detach().numpy()
    # out_torch,zx = model_torch(x_torch,t_torch,y_torch).detach().numpy()
    # out_torch=out_torch.detach().numpy()

    # out_torch = einops.rearrange(out_torch, 'b c h w -> b h w c')

    print(out_torch.shape,out_torch.max())
    print(out_jax.shape,out_jax.max())
    print(np.max(np.abs(out_jax - out_torch)))
    np.testing.assert_almost_equal(out_jax, out_torch, decimal=5)







# def test_convert_mlp():
#     shape = (1, 3, 224, 224)
#     x_torch = torch.rand(shape)
#     x_jax = jnp.array(x_torch.detach().numpy())
#     x_jax = einops.rearrange(x_jax, 'b c h w -> b h w c')
#
#     model_kwargs = {
#         'hidden_size': 3,
#         'projector_dim': 10,
#         'z_dim':100
#         # 'norm_layer':None
#
#     }
#
#     rngs = jax.random.PRNGKey(1)
#
#     model_torch = build_mlp(**model_kwargs,)
#     params_torch = {k: v.numpy() for k, v in model_torch.state_dict().items()}
#     params_torch = flax.traverse_util.unflatten_dict(params_torch, sep=".")
#
#     model_kwargs.pop('in_channels')
#     model_jax = MLP(**model_kwargs)
#     params_jax = model_jax.init(rngs, x_jax)['params']
#     print(model_torch.state_dict().keys())
#
#     def test_print(p, x):
#         print(p)
#
#     # jax.tree_util.tree_map_with_path(test_print,params_jax)
#     params_jax = convert_torch_to_flax_mlp(params_torch)
#
#     out_jax = model_jax.apply({'params': params_jax}, x_jax)
#     out_jax = np.array(out_jax)
#     out_torch = model_torch(x_torch).detach().numpy()
#     out_torch = einops.rearrange(out_torch, 'b c h w -> b h w c')
#
#     print(out_torch.shape)
#     print(out_jax.shape)
#     print(np.max(np.abs(out_jax - out_torch)))
#     np.testing.assert_almost_equal(out_jax, out_torch, decimal=6)



if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'cpu')
    # test_convert_mlp()
    # test_convert_temb()
    # test_convert_lemb()
    # test_convert_patch_embed()
    # test_convert_sit_block()
    test_convert_sit()