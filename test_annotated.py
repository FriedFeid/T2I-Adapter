from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
import torch
from diffusers import LMSDiscreteScheduler
import glob
import numpy as np
from PIL import Image
import control_utils as cu


def get_imgs(image_root):

    paths = glob.glob(image_root+'*_depth*.png')

    instances = []
    for im_path in paths:
        images = {}
        name = im_path.split('/')[-1]
        name, alternative = name.split('_depth')[0], name.split('_depth')[1]
        if '_alt' in alternative:
            alternative = True
        else:
            alternative = False

        image = cu.get_image(image_root + name + '.png')
        depth = Image.open(im_path)
        if not depth.mode == 'L':
            depth = depth.convert("L")
        images['original'] = image
        images['depth'] = np.array(depth).astype(np.float32)/255.
        images['Prompt'] = name
        images['alt'] = alternative
        if 'badPrompts' in image_root or 'scales' in image_root:
            if 'cube on' in name:
                images['insuf_Prompt'] = 'render of a lavender sphere floating in the air'
                images['conflicting_Prompt'] = 'image of a house surrounded by an beautiful garden'
            else:
                images['insuf_Prompt'] = 'high quality, 4k, detailed, professional work'
                images['conflicting_Prompt'] = 'high quality photo of a delecious cake'
        # print(np.min(images['depth']), np.max(images['depth']), images['depth'].shape)
        instances.append(images)
        return instances


if __name__ == '__main__':
    # load adapter
    adapter = T2IAdapter.from_pretrained(
      "/export/data/vislearn/rother_subgroup/feiden/models/pretrained/T2I_Adapter/midas/diffusion_pytorch_model.fp16(1).safetensors",
      torch_dtype=torch.float16).to("cuda")

    # load euler_a scheduler
    model_id = '/export/data/vislearn/rother_subgroup/dzavadsk/models/pretrained_originals/SDXL/sd_xl_base_1.0_0.9vae.safetensors'
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder=False)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
    ).to("cuda")
    scheduler = LMSDiscreteScheduler.from_config('/export/data/vislearn/rother_subgroup/feiden/models/pretrained/T2I_Adapter/midas/config(1).json')
    pipe.scheduler = scheduler


    pipe.enable_xformers_memory_efficient_attention()

    instances = get_imgs('/export/data/ffeiden/PaperControlnetXS/512_images/')
    curr = instances[8]

    prompt = curr['Prompt']
    negative_prompt = ""
    gen_images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=curr['depht'],
        num_inference_steps=50,
        adapter_conditioning_scale=0.8,
        guidance_scale=7.5,
    ).images[0]
    gen_images.save('out_lin.png')
