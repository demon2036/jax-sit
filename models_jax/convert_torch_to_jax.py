import jax


def convert_torch_to_flax_linear(torch_params, prefix='', sep=''):
    state_dict = {
        f'{prefix}{sep}kernel': torch_params['weight'].T,  # Transpose for correct weight shape
    }
    if 'bias' in torch_params:
        state_dict[f'{prefix}{sep}bias'] = torch_params['bias']
    return state_dict



def convert_torch_to_flax_mlp(torch_params, prefix='', sep=''):
    state_dict=dict()
    state_dict['0']=convert_torch_to_flax_linear(torch_params['0'])
    state_dict['2'] = convert_torch_to_flax_linear(torch_params['2'])
    state_dict['4'] = convert_torch_to_flax_linear(torch_params['4'])
    return state_dict



def convert_torch_to_flax_temb(torch_params, prefix='', sep=''):
    torch_params=torch_params['mlp']
    state_dict=dict()
    state_dict['0']=convert_torch_to_flax_linear(torch_params['0'])
    state_dict['2'] = convert_torch_to_flax_linear(torch_params['2'])
    return state_dict


def convert_torch_to_flax_lemb(torch_params, prefix='', sep=''):
    state_dict=dict()
    state_dict['embedding']=dict()

    state_dict['embedding']['embedding']=torch_params['embedding_table']['weight']

    return state_dict




def convert_torch_to_flax_attention(torch_params, prefix='', sep=''):
    flax_params = {
        f'{prefix}{sep}qkv': convert_torch_to_flax_linear(torch_params['qkv']  ),
        f'{prefix}{sep}proj': convert_torch_to_flax_linear(torch_params['proj']),
    }
    return flax_params



def convert_torch_to_flax_conv(torch_params, prefix='', sep=''):
    state_dict = {
        f'{prefix}{sep}kernel': torch_params['weight'].transpose(2, 3, 1, 0),  # PyTorch: [O, I, H, W] -> Flax: [H, W, I, O]
    }

    if 'bias' in torch_params:
        state_dict[f'{prefix}{sep}bias'] = torch_params['bias']  # Copy bias directly

    return state_dict


def convert_torch_to_flax_patch_embed(torch_params, prefix='', sep=''):
    state_dict={
        'proj':convert_torch_to_flax_conv(torch_params['proj'])
    }
    return state_dict



def convert_torch_to_flax_mlp_sit_block(torch_params, prefix='', sep=''):
    state_dict=dict()
    state_dict['fc1']=convert_torch_to_flax_linear(torch_params['fc1'])
    state_dict['fc2'] = convert_torch_to_flax_linear(torch_params['fc2'])
    return state_dict

def convert_torch_to_flax_sit_block(torch_params, prefix='', sep=''):
    state_dict=dict()
    state_dict['attn']=convert_torch_to_flax_attention(torch_params['attn'])
    state_dict['mlp']=convert_torch_to_flax_mlp_sit_block(torch_params['mlp'])
    state_dict['Dense_0']=convert_torch_to_flax_linear(torch_params['adaLN_modulation']['1'])
    return state_dict


def convert_torch_to_flax_final_layer(torch_params, prefix='', sep=''):

    state_dict=dict()
    state_dict['proj']=convert_torch_to_flax_linear(torch_params['linear'])
    state_dict['ada_dense']=convert_torch_to_flax_linear(torch_params['adaLN_modulation']['1'])
    return state_dict




def convert_torch_to_flax_sit(torch_params, prefix='', sep=''):
    print(torch_params.keys())
    state_dict={'pos_embed':torch_params['pos_embed']}
    state_dict['t_embedder']=convert_torch_to_flax_temb(torch_params['t_embedder'])
    state_dict['y_embedder']=convert_torch_to_flax_lemb(torch_params['y_embedder'])
    state_dict['x_embedder']=convert_torch_to_flax_patch_embed(torch_params['x_embedder'])
    state_dict['final_layer'] = convert_torch_to_flax_final_layer(torch_params['final_layer'])



    i=0
    blocks=torch_params['blocks']
    while f'{i}' in blocks:
        state_dict[f'blocks_{i}']=convert_torch_to_flax_sit_block(blocks[f'{i}'])
        i+=1



    # while 1:
    #     1
    return state_dict





