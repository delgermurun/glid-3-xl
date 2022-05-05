import os.path
from pathlib import Path

import clip
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from .cli_parser import static_args
from .encoders.modules import BERTEmbedder
from .guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(
                torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
            )
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety: offsety + size, offsetx: offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


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
        model_params['timestep_respacing'] = 'ddim' + str(static_args.steps)
    else:
        model_params['timestep_respacing'] = 'ddim50'
elif static_args.steps:
    model_params['timestep_respacing'] = str(static_args.steps)

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

# clip
clip_model, clip_preprocess = clip.load('ViT-L/14', device=device, jit=False)
clip_model.eval().requires_grad_(False)
normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)


def do_run(runtime_args):
    if runtime_args.seed >= 0:
        torch.manual_seed(runtime_args.seed)

    # bert context
    text_emb = (
        bert.encode([runtime_args.text] * runtime_args.batch_size).to(device).float()
    )
    text_blank = (
        bert.encode([runtime_args.negative] * runtime_args.batch_size)
            .to(device)
            .float()
    )

    text = clip.tokenize(
        [runtime_args.text] * runtime_args.batch_size, truncate=True
    ).to(device)
    text_clip_blank = clip.tokenize(
        [runtime_args.negative] * runtime_args.batch_size, truncate=True
    ).to(device)

    # clip context
    text_emb_clip = clip_model.encode_text(text)
    text_emb_clip_blank = clip_model.encode_text(text_clip_blank)

    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, runtime_args.cutn)

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
        "clip_embed": torch.cat([text_emb_clip, text_emb_clip_blank], dim=0).float()
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

    cur_t = None

    def cond_fn(x, t, context=None, clip_embed=None, image_embed=None):
        with torch.enable_grad():
            x = x[: runtime_args.batch_size].detach().requires_grad_()

            n = x.shape[0]

            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t

            kw = {
                'context': context[: runtime_args.batch_size],
                'clip_embed': clip_embed[: runtime_args.batch_size]
                if model_params['clip_embed_dim']
                else None,
                'image_embed': image_embed[: runtime_args.batch_size]
                if image_embed is not None
                else None,
            }

            out = diffusion.p_mean_variance(
                model, x, my_t, clip_denoised=False, model_kwargs=kw
            )

            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)

            x_in /= 0.18215

            x_img = ldm.decode(x_in)

            clip_in = normalize(make_cutouts(x_img.add(1).div(2)))
            clip_embeds = clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(
                clip_embeds.unsqueeze(1), text_emb_clip.unsqueeze(0)
            )
            dists = dists.view([runtime_args.cutn, n, -1])

            losses = dists.sum(2).mean(0)

            loss = losses.sum() * runtime_args.clip_guidance_scale

            return -torch.autograd.grad(loss, x)[0]

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

            filename = os.path.join(runtime_args.output_path,
                f'{runtime_args.prefix}{i * runtime_args.batch_size + k:05}.png'
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


    print(init)

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
            cond_fn=cond_fn if runtime_args.clip_guidance else None,
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
