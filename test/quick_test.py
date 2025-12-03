import torch
from torchvision.transforms import v2 as T
from torchvision.io import read_image
import yaml

import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def main(args):

    if args.model == 'IDRWKV':
        from arch.idrwkv import IDRWKV
    elif args.model == 'IDRWKVS':
        from arch.dirwkvs import IDRWKV

    if args.model == 'IDRWKV':
         config = f'train_yamls/{args.scale}X_IDRWKV_ft.yaml'
    elif args.model == 'IDRWKVS':
        config = f'train_yamls/{args.scale}X_IDRWKVS.yaml'
    with open(config, 'r') as f:
        config = yaml.safe_load(f)


    if args.model == 'IDRWKV':
        state_dict = torch.load(f'checkpoints/{args.scale}X_IDRWKV/iteration_{args.iter}.pt')
    elif args.model == 'IDRWKVS':
        state_dict = torch.load(f'checkpoints/{args.scale}X_IDRWKVS_3/iteration_{args.iter}.pt')
    iteration = state_dict['iteration']
    model_config = config['model']

    model = IDRWKV(
        img_size=model_config['patch_size'],
        num_blocks=model_config['num_blocks'],
        hidden_rate=model_config['hidden_rate'],
        patch_size=model_config['patch_size'],
        img_range=1,
        embed_dim=model_config['embed_dim'],
        upscale=model_config['scale'],
        n_head=model_config['num_heads']
    ).to(device='cuda')
    model.load_state_dict(state_dict['model_state_dict'], strict=False)
    model.eval()

    # img = None
    # if args.img:
    #     img = read_image(f'training_data/{args.img}.png')
    # else: 
    #     img = read_image('training_data/test.png')
    img_name = args.img if args.img else '92_4x'
    image_path = f'training_data/{img_name}.png'
    img = read_image(image_path)



    img = img[:3, :, :]
    img = T.ToDtype(torch.float32, scale=True)(img)
    img = img.unsqueeze(0).to(device='cuda')
    original_shape = img.shape

    pred = model(img)

    output_shape = pred.shape

    to_pil = T.ToPILImage()
    final = to_pil(pred.squeeze(0))
    final.save(f'test_outputs/{img_name}_{args.scale}X_{args.model}_iter_{args.iter}.png')

    print(f'iteration evaluated: {iteration} | input dim = {original_shape} | output dim = {output_shape} | scale = {args.scale}')
    print(f'saved to test_outputs/{img_name}_{args.scale}X_{args.model}_iter_{args.iter}.png')

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, required=True, help='scale')
parser.add_argument('--iter', type=int, required=True, help='iter')
parser.add_argument('--img', type=str, required=False, help='img to test')
parser.add_argument('--model', type=str, required=False, default='IDRWKV', help='model architecture')
args = parser.parse_args()

if __name__ == '__main__':
    main(args)
