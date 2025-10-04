import torch
from torchvision.transforms import v2 as T
from torchvision.io import read_image
import yaml


import sys
import os
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from arch.rwkv6sr import RWKVIR

def main(args):
    config = 'train_yamls/RWVKSR_2X.yaml'
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    state_dict = torch.load(f'checkpoints/{args.scale}X_RWKV6_{args.version}SR/iteration_{args.iter}.pt')
    iteration = state_dict['iteration']
    model_config = config['model']
    depths = [model_config['blocks_per_layer']] * model_config['residual_groups']
    model = RWKVIR(
        img_size=model_config['patch_size'],
        depths=depths,
        mlp_ratio=3.,
        patch_size=model_config['patch_size'],
        img_range=1,
        embed_dim=model_config['embed_dim'],
        upscale=model_config['scale'],
        upsampler='pixelshuffledirect',
        resi_connection=model_config['resi_connection'],
        n_head=model_config['num_heads']
    ).to(device='cuda')
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    img = None
    if args.img:
        img = read_image(f'training_data/{args.img}.png')
    else: 
        img = read_image('training_data/test.png')

    img = img[:3, :, :]
    img = T.ToDtype(torch.float32, scale=True)(img)
    img = img.unsqueeze(0).to(device='cuda')
    original_shape = img.shape

    pred = model(img)

    output_shape = pred.shape

    to_pil = T.ToPILImage()
    final = to_pil(pred.squeeze(0))
    final.save(f'test_outputs/iteration_{iteration}_{args.version}.png')

    print(f'iteration evaluated: {iteration} | input dim = {original_shape} | output dim = {output_shape} | scale = {args.scale}')

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, required=True, help='scale')
parser.add_argument('--iter', type=int, required=True, help='iter')
parser.add_argument('--img', type=str, required=False, help='img to test')
parser.add_argument('--version', type=str, required=True, help='version')
args = parser.parse_args()

if __name__ == '__main__':
    main(args)
