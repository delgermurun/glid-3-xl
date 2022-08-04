import os.path
from pathlib import Path
import json

import requests
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .cli_parser import static_args
from .encoders.modules import BERTEmbedder
from .guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

device = torch.device(
    'cuda:0' if (torch.cuda.is_available() and not static_args.cpu) else 'cpu'
)
print('Using device:', device)

model_state_dict = torch.load(static_args.model_path, map_location='cpu')

model_params = {
    'attention_resolutions': '32,16,8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '27',  # Modify this value to decrease the number of
    # timesteps.
    'image_size': 32,
    'learn_sigma': False,
    'noise_schedule': 'linear',
    'num_channels': 320,
    'num_heads': 8,
    'num_res_blocks': 2,
    'resblock_updown': False,
    'use_fp16': False,
    'use_scale_shift_norm': False,
    'clip_embed_dim': 768 if 'clip_proj.weight' in model_state_dict else None,
    'image_condition': True
    if model_state_dict['input_blocks.0.0.weight'].shape[1] == 8
    else False,
    'super_res_condition': True
    if 'external_block.0.0.weight' in model_state_dict
    else False,
}

if static_args.ddpm:
    model_params['timestep_respacing'] = 1000
if static_args.ddim:
    if static_args.steps:
        model_params['timestep_respacing'] = 'ddim' + os.environ.get(
            'GLID3_STEPS', str(static_args.steps)
        )
    else:
        model_params['timestep_respacing'] = 'ddim50'
elif static_args.steps:
    model_params['timestep_respacing'] = os.environ.get(
        'GLID3_STEPS', str(static_args.steps)
    )

model_config = model_and_diffusion_defaults()
model_config.update(model_params)

if static_args.cpu:
    model_config['use_fp16'] = False

# Load models
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(model_state_dict, strict=False)
model.requires_grad_(static_args.clip_guidance).eval().to(device)

if model_config['use_fp16']:
    model.convert_to_fp16()
else:
    model.convert_to_fp32()


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


# vae
ldm = torch.load(static_args.kl_path, map_location="cpu")
ldm.to(device)
ldm.eval()
ldm.requires_grad_(static_args.clip_guidance)
set_requires_grad(ldm, static_args.clip_guidance)

bert = BERTEmbedder(1280, 32)
sd = torch.load(static_args.bert_path, map_location="cpu")
bert.load_state_dict(sd)

bert.to(device)
bert.half().eval()
set_requires_grad(bert, False)

blank_clip_embedding = None
blank_bert_embedding = None

def do_run(runtime_args, text_emb_clip):
    global blank_clip_embedding, blank_bert_embedding

    if runtime_args.seed >= 0:
        torch.manual_seed(runtime_args.seed)

    if blank_bert_embedding is None:
        print('Generating blank BERT embedding')
        blank_bert_embedding = (
            bert.encode([runtime_args.negative])
        ).to(device).float()

    if blank_clip_embedding is None:
        print('Generating blank CLIP embedding')
        res = requests.post('https://demo-cas.jina.ai:8443/post',
            data=json.dumps({'execEndpoint':'/', 'data': [{'text': runtime_args.negative}]}),
            headers={'content-type': 'application/json'}
        ).json()

        blank_clip_embedding = np.array(res['data'][0]['embedding']).astype(
            model_config['use_fp16'] and np.float16 or np.float32
        ).reshape(1, -1)

    # bert context
    text_emb = (
        bert.encode([runtime_args.text]).to(device).float()
    ).repeat(runtime_args.batch_size, 1, 1)
    text_blank = blank_bert_embedding.repeat(runtime_args.batch_size, 1, 1)

    # clip context
    text_emb_clip = np.repeat(text_emb_clip[np.newaxis, :], runtime_args.batch_size, axis=0)
    text_emb_clip_blank = np.repeat(blank_clip_embedding, runtime_args.batch_size, axis=0)

    # torch.Size([8, 77, 1280]) torch.Size([8, 77, 1280]) (1, 768) (1, 768)

    print(text_emb_clip.shape, type(text_emb_clip))
    image_embed = None

    # image context
    if model_params['image_condition']:
        # using inpaint model but no image is provided
        image_embed = torch.zeros(
            runtime_args.batch_size * 2,
            4,
            runtime_args.height // 8,
            runtime_args.width // 8,
            device=device,
        )

    kwargs = {
        "context": torch.cat([text_emb, text_blank], dim=0).float(),
        "clip_embed": torch.cat(
            [torch.from_numpy(text_emb_clip), torch.from_numpy(text_emb_clip_blank)],
            dim=0,
        )
        .to(device)
        .float()
        if model_params['clip_embed_dim']
        else None,
        "image_embed": image_embed,
    }

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + runtime_args.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    if runtime_args.ddpm:
        sample_fn = diffusion.ddpm_sample_loop_progressive
    elif runtime_args.ddim:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    def save_sample(i, sample):
        for k, image in enumerate(sample['pred_xstart'][: runtime_args.batch_size]):
            image /= 0.18215
            im = image.unsqueeze(0)
            out = ldm.decode(im)

            out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))

            Path(runtime_args.output_path).mkdir(exist_ok=True)

            filename = os.path.join(
                runtime_args.output_path,
                f'{runtime_args.prefix}{i * runtime_args.batch_size + k:05}.png',
            )
            out.save(filename)

    if runtime_args.init_image:
        init = Image.open(runtime_args.init_image).convert('RGB')
        init = init.resize(
            (int(runtime_args.width), int(runtime_args.height)), Image.LANCZOS
        )
        init = TF.to_tensor(init).to(device).unsqueeze(0).clamp(0, 1)
        h = ldm.encode(init * 2 - 1).sample() * 0.18215
        init = torch.cat(runtime_args.batch_size * 2 * [h], dim=0)
    else:
        init = None

    for i in range(runtime_args.num_batches):
        cur_t = diffusion.num_timesteps - 1

        samples = sample_fn(
            model_fn,
            (
                runtime_args.batch_size * 2,
                4,
                int(runtime_args.height / 8),
                int(runtime_args.width / 8),
            ),
            clip_denoised=False,
            model_kwargs=kwargs,
            cond_fn=None,
            device=device,
            progress=True,
            init_image=init,
            skip_timesteps=runtime_args.skip_timesteps,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % 5 == 0 and j != diffusion.num_timesteps - 1:
                save_sample(i, sample)

        save_sample(i, sample)
