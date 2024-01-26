import ldm.data.laionAE as lae
import numpy as np
import os
from pytorch_lightning import seed_everything
import torch
import yaml
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, AutoencoderKL, DDIMScheduler
from controlnet_aux.midas import MidasDetector
from PIL import Image

torch.set_grad_enabled(False)


def create_sd_sample_set(
        pipe,
        ds,
        path2samples,
        model_version,
        n_samples=None,
        scale=9.5,
        eta=0.5,
        ddim_steps=25,
        caption_idx=0,
        control_scale=1.0,
        control_mode='midas',
):

    idx = 0
    seed_everything(42)
    sample_save_path = os.path.join(path2samples, model_version,
                                    f'steps-{ddim_steps}',
                                    f'caption-{caption_idx}'
                                    )
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)
        print(f'[CREATED \n{sample_save_path} \n]')

    if control_mode == 'midas':
        midas_depth = MidasDetector.from_pretrained(
            "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large",
            cache_dir='/export/scratch/ffeiden/Pretrained%20Originals/T2I_XL/'
            ).to("cuda")

    while idx < min(len(ds), n_samples):

        if control_mode == 'canny':
            image = torch.from_numpy(ds[idx]['hint'].copy()).float().permute(2, 0, 1)

        elif control_mode == 'midas':
            image = Image.fromarray(((ds[idx]['image'] / 2. + 0.5) * 255.).astype(np.uint8))

            if not image.mode == "RGB":
                image = image.convert("RGB")

            image = midas_depth(
                image, detect_resolution=512, image_resolution=512
                )

        prompt = ds[idx]['caption']
        negative_prompt = ''

        #### sampling

        gen_images = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image,
                        num_inference_steps=ddim_steps,
                        adapter_conditioning_scale=control_scale,
                        guidance_scale=scale,
                        ).images[0]

        gen_images.save(os.path.join(sample_save_path, f'{idx:06}.jpg'))

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


n_samples = 5_000
cfg = 9.5  # classifier free guidance scale
eta = 0.5
ddim_steps = 50
caption_idx = 2
control_scale = 1
control_mode = 'midas'  # ('canny', 'midas')
path2samples = '/export/data/ffeiden/ResultsControlNetXS/T2I/'  # root for samples
model_version = 't2i_SDXL_depth'  # name of model/version
sdxl = False

# load adapter
adapter = T2IAdapter.from_pretrained(
  "TencentARC/t2i-adapter-depth-midas-sdxl-1.0", torch_dtype=torch.float16,
  cache_dir='/export/scratch/ffeiden/Pretrained%20Originals/T2I_XL/'
).to("cuda")

# load euler_a scheduler
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
euler_a = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler",
                                        cache_dir='/export/scratch/ffeiden/Pretrained%20Originals/T2I_XL/')
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
                                    cache_dir='/export/scratch/ffeiden/Pretrained%20Originals/T2I_XL/')
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16,
    cache_dir='/export/scratch/ffeiden/Pretrained%20Originals/T2I_XL/'
    ).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

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
        pipe=pipe,
        ds=coco_set,
        path2samples=path2samples,
        model_version=model_version,
        n_samples=n_samples,
        scale=cfg,
        eta=eta,
        ddim_steps=ddim_steps,
        caption_idx=caption_idx,
        control_scale=control_scale,
        control_mode=control_mode,
    )
