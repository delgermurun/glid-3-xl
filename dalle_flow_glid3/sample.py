import os.path
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .model import bert, ldm, diffusion, model_params, device, model

def do_run(runtime_args, text_emb_clip, blank_bert_embedding, blank_clip_embedding):
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
