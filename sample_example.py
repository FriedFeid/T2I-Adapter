import ldm.data.laionAE as lae
import numpy as np
from ldm.models.diffusion.ddim import DDIMSampler
import copy
import os
from pytorch_lightning import seed_everything
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector

from torchvision import transforms as tt
import torch
import yaml
import os
import cv2
import torch
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)

torch.set_grad_enabled(False)


class deafult_settings():
    def __init__(self):
        self.outdir = '/export/home/ffeiden/Projects/T2I-Adapter/outputs/test_gen' # str
        self.prompt = ''  # str
        self.neg_prompt = ''  # str longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                              # 'fewer digits, cropped, worst quality, low quality
        self.cond_path = ''  # str: condition image path
        self.cond_inp_type = 'depth'  # str: the type of the input condition image, take depth T2I as example, the input can be raw image, '
                                      # 'which depth will be calculated, or the input can be a directly a depth map image
        self.sampler = 'ddim'  # str: ddim, plms
        self.steps = 50  # int: numper of sampling steps
        self.sd_ckpt = '/export/data/vislearn/rother_subgroup/dzavadsk/models/pretrained_originals/StableDiffusion/v1-5-pruned.ckpt'
                         # str: path to sd ckpt or safetensor
        self.vae_ckpt = None  # str: VAE checkpoint
        self.adapter_ckpt = '/export/data/vislearn/rother_subgroup/feiden/models/pretrained/T2I_Adapter/t2iadapter_depth_sd15v2.pth'
                             # str: Adapter ckpt
        self.config = 'configs/stable-diffusion/sd-v1-inference.yaml' # str: path to config
        self.max_resolution = 512 * 512  # float, max image hight * width
        self.resize_short_edge = None  # int: resize short edge of the input image, if set max_res not used
        self.C = 4  # int: latent channels
        self.f = 8  # int: downsampling factor
        self.scale = 7.5  # float: unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
        self.cond_tau = 1.0  # float: timestamp parameter that determines until which step adapter is applied
        self.style_cond_tau = 1.0  # timestamp parameter that determines until which step the adapter is applied
        self.cond_weight = 1.0  # float: the adapter features are multiplied with this (control strength)
        self.seed = 42  # int
        self.n_samples = 1  # int:  # of samples to generate
        self.which_cond = 'depth'  #str:  sketch keypose seg depth canny style color openpose
        self.device = 'cuda'  # str: cuda


opt = deafult_settings()


def create_sd_sample_set(
        sd_model,
        process_cond_module,
        adapter,
        ds,
        path2samples,
        model_version,
        sampler,
        n_samples=None,
        scale=9.5,
        eta=0.5,
        shape=(4, 64, 64),
        ddim_steps=25,
        bs=5,
        caption_idx=0,
        control_scale=1.0,
        control_mode='canny',
):

    sd_model, sampler = get_sd_models(opt)
    adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))

    sample_save_path = os.path.join(path2samples, model_version, f'steps-{ddim_steps}', f'caption-{caption_idx}')
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)
        print(f'[CREATED \n{sample_save_path} \n]')

    idx = 0
    seed_everything(42)
    if control_mode == 'midas':
        midas = MidasDetector()

    while idx < min(len(ds), n_samples):
        hints = None
        prompts = None

        if control_mode == 'canny':
            hints = torch.from_numpy(ds[idx]['hint'].copy()).float().permute(2, 0, 1)

        elif control_mode == 'midas':
            new = (ds[idx]['image'] / 2. + 1) * 255.
            hint, _ = midas(np.array(new))
            hints = hint

        prompts = ds[idx]['caption']

        #### sampling

        with torch.inference_mode(), \
                    sd_model.ema_scope(), \
                    autocast('cuda'):
            cond_model = None
            if opt.cond_inp_type == 'image':
                cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))
            cond = process_cond_module(opt, hints, opt.cond_inp_type, cond_model)

            adapter_features, append_to_context = get_adapter_feature(cond, adapter)
            opt.prompt = prompts
            x_samples = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context,
                                            batch_size=1
                                            )

        ####
        cv2.imwrite(os.path.join(sample_save_path, f'{idx:06}.jpg'), tensor2img(x_samples))

        if idx >= min(len(ds), n_samples):
            break
        else:
            idx += 1

    stats_dict = {
        'cfg_scale': scale,
        'eta': eta,
        'control_scale': control_scale,
        'ddim_steps': ddim_steps,
        'caption_idx': caption_idx,
        'sample_path': sample_save_path
    }

    with open(os.path.join(sample_save_path, 'stats_dict.yaml'), 'w') as f:
        yaml.dump(stats_dict, f, default_flow_style=False)

    print(f'[SAMPLES CALCULATED AND SAVED IN\n{sample_save_path}]')


# prepare models
which_cond = opt.which_cond

sd_model, sampler = get_sd_models(opt)
adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))

process_cond_module = getattr(api, f'get_cond_{which_cond}')
##

n_samples = 5_000
cfg = 9.5  # classifier free guidance scale
eta = 0.5
ddim_steps = 50
caption_idx = 2
control_scale = 1.0
control_mode = 'midas'  # ('canny', 'midas')
path2samples = '/export/data/ffeiden/ResultsControlNetXS/T2I/'  # root for samples
model_version = 't2i_depth'  # name of model/version
sdxl = False


caption_csv_list = [
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val_idx0.json',
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val_idx1.json',
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val_idx2.json',
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val_idx3.json',
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val_idx4.json',
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val.json'
                    ]
data_csv = '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_list_val.txt'
caption_csv = caption_csv_list[caption_idx]

coco_set = lae.LaionBase(
    size=512,
    random_resized_crop=False,
    control_mode='canny',
    data_root='/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/val2017',
    full_set=True,
    data_csv=data_csv,
    caption_csv=caption_csv,
    np_format=not sdxl,
    original_size_as_tuple=True,
    crop_coords_top_left=True,
    target_size_as_tuple=True,
)

np.random.seed(42)
coco_set.canny_tresholds = np.concatenate([
    np.random.randint(50, 100, [len(coco_set), 1]),
    np.random.randint(200, 350, [len(coco_set), 1])], axis=1
    )

create_sd_sample_set(
    sd_model=sd_model,
    process_cond_module=process_cond_module,
    adapter=adapter,
    path2samples=path2samples,
    model_version=model_version,
    sampler=sampler,
    ds=coco_set,
    n_samples=n_samples,
    scale=cfg,
    eta=eta,
    shape=(4, 64, 64),
    ddim_steps=ddim_steps,
    bs=5,
    caption_idx=caption_idx if caption_idx >= 0 else 'all',
    control_scale=control_scale,
    control_mode=control_mode,
    )
