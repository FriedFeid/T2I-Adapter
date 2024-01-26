from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, AutoencoderKL, DDIMScheduler
import torch
from pytorch_lightning import seed_everything
import glob
import numpy as np
from PIL import Image
import control_utils as cu
import ldm.data.laionAE as lae
from controlnet_aux.canny import CannyDetector


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


# load adapter
adapter = T2IAdapter.from_pretrained(
  "TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16,
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

# image = np.array(image)
# image = image[:512, :512, :]
# image = Image.fromarray(image)
# if not image.mode == "RGB":
#     image = image.convert("RGB")

# prompt = "A photo of a room, 4k photo, highly detailed"
# negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"

caption_csv_list = [
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val_idx0.json',
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val_idx1.json',
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val_idx2.json',
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val_idx3.json',
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val_idx4.json',
    '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_captions_val.json'
                    ]
data_csv = '/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/coco2017_image_list_val.txt'
caption_csv = caption_csv_list[2]
sdxl = False

coco_set = lae.LaionBase(
    size=512,
    random_resized_crop=False,
    control_mode='canny',
    data_root='/export/data/vislearn/rother_subgroup/dzavadsk/datasets/coco2017/val2017',
    full_set=True,
    data_csv=data_csv,
    caption_csv=caption_csv,
    np_format=True,
    original_size_as_tuple=True,
    crop_coords_top_left=True,
    target_size_as_tuple=True,
)
image = Image.fromarray(((coco_set[0]['image'] / 2. + 0.5) * 255.).astype(np.uint8))


if not image.mode == "RGB":
    image = image.convert("RGB")

canny_detector = CannyDetector()
image = canny_detector(image, low_threshold=100, high_threshold=200)
prompt = coco_set[0]['caption']
image.save('edges.png')


# instances = get_imgs('/export/data/ffeiden/PaperControlnetXS/512_images/')
# curr = instances[6]

# prompt = curr['Prompt']
negative_prompt = ""
# image = Image.fromarray((curr['depth'] * 255.).astype(np.uint8))
# if not image.mode == "RGB":
#     image = image.convert("RGB")

seed_everything(1995)

gen_images = pipe(
  prompt=prompt,
  negative_prompt=negative_prompt,
  image=image,
  num_inference_steps=30,
  adapter_conditioning_scale=1,
  guidance_scale=7.5,
).images[0]
gen_images.save('out_mid_wide.png')
