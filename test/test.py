import os
from common.test_utils import *
import torch
import yaml
import argparse
# from arch.rwkv4srlite3 import RWKVIR
from arch.rwkv6srlite2 import RWKVIR
from torchvision.transforms import v2 as T
from torchvision.io import read_image
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def tensor_to_uint8(img_tensor):
    """
    Expect tensor in [0,1] (float32/float16), shape C×H×W or H×W×C.
    Returns np.uint8 array in [0,255].
    """
    img = img_tensor.detach().cpu()
    if img.dim() == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)  # CHW → HWC
    img = img.clamp(0, 1).mul(255).round()
    img = img.numpy().astype(np.uint8)
    return img[..., ::-1]


def test():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--iter', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # get all image pairs
    lr_test_folder = config['lr_folder']
    hr_test_folder = config['hr_folder']

    dataset_names = os.listdir(lr_test_folder)
    scales = config['scales']

    dataset_pairs = {}


    
    for dataset_name in dataset_names:
        for scale in scales:
            lr_folder = os.path.join(lr_test_folder, dataset_name, scale)
            hr_folder = os.path.join(hr_test_folder, dataset_name, scale)
            lr_images = sorted(os.listdir(lr_folder))
            hr_images = sorted(os.listdir(hr_folder))
            image_pairs = []
            for lr_image in lr_images:
                hr_img_name = lr_image.replace(f'_LRBI_', '_HR_')
                if hr_img_name in hr_images:
                    image_pairs.append((os.path.join(lr_folder, lr_image), os.path.join(hr_folder, hr_img_name)))
            dataset_pairs[f'{dataset_name}_{scale}'] = image_pairs
    
    # print(dataset_pairs['Set5_x2'])
    device = 'cuda'
    model_config = config['model']
    depths = [model_config['blocks_per_layer']] * model_config['residual_groups']
    # model = RWKVIR(
    #     img_size=model_config['patch_size'],
    #     depths=depths,
    #     hidden_rate=model_config['hidden_rate'],
    #     patch_size=model_config['patch_size'],
    #     embed_dim=model_config['embed_dim'],
    #     upscale=model_config['scale'],
    #     upsampler=model_config['upsampler'],
    #     resi_connection=model_config['resi_connection']
    # )
    model = RWKVIR(
        img_size=model_config['patch_size'],
        depths=depths,
        hidden_rate=model_config['hidden_rate'],
        patch_size=model_config['patch_size'],
        img_range=1,
        embed_dim=model_config['embed_dim'],
        upscale=model_config['scale'],
        n_head=model_config['num_heads']
    )
    model = model.to(device)
    checkpoint = torch.load(os.path.join(config['checkpoint_folder'], f'iteration_{args.iter}.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    ssims = []
    psnrs = []
    for dataset_scale in dataset_pairs.keys():
        image_pairs = dataset_pairs[dataset_scale]
        current_scale = int(dataset_scale.split('_')[-1].replace('x', ''))
        # (lr, hr)
        dataset_ssims = []
        dataset_psnrs = []
        start = time.time()
        for image_pair in image_pairs:
            lr = read_image(image_pair[0])
            lr = T.ToDtype(torch.float32, scale=True)(lr)
            lr = lr.unsqueeze(0)  # add batch dimension
            lr = lr.to(device)

            hr = read_image(image_pair[1])
            hr = T.ToDtype(torch.float32, scale=True)(hr)
        

            with torch.no_grad():
                predicted = model(lr)
                predicted = predicted.squeeze(0)

            predicted_arr = tensor_to_uint8(predicted)
            hr_arr = tensor_to_uint8(hr)

            ssim = calculate_ssim(predicted_arr, hr_arr, crop_border=current_scale, input_order='HWC', test_y_channel=True)
            psnr = calculate_psnr(predicted_arr, hr_arr, crop_border=current_scale, input_order='HWC', test_y_channel=True)

            dataset_ssims.append(ssim)
            dataset_psnrs.append(psnr)
        end = time.time()

        avg_ssim = sum(dataset_ssims) / len(dataset_ssims)
        avg_psnr = sum(dataset_psnrs) / len(dataset_psnrs)

        print(f'Dataset: {dataset_scale}, PSNR: {avg_psnr:.4f},  SSIM: {avg_ssim:.4f}, Time: {end - start:.2f}s')
        ssims.append(avg_ssim)
        psnrs.append(avg_psnr)

            

            


        


if __name__ == '__main__':
    test()