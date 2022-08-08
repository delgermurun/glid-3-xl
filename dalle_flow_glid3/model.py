import os.path

import torch

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