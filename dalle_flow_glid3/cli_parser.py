# argument parsing
import argparse
import os

parser = argparse.ArgumentParser()

glid_path = os.environ.get('GLID_MODEL_PATH', '')

parser.add_argument(
    '--model_path',
    type=str,
    default=f'{glid_path}/finetune.pt',
    help='path to the diffusion model',
)

parser.add_argument(
    '--kl_path',
    type=str,
    default=f'{glid_path}/kl-f8.pt',
    help='path to the LDM first stage model',
)

parser.add_argument(
    '--bert_path',
    type=str,
    default=f'{glid_path}/bert.pt',
    help='path to the LDM first stage model',
)

parser.add_argument(
    '--text', type=str, required=False, default='', help='your text prompt'
)

parser.add_argument(
    '--negative', type=str, required=False, default='a', help='negative text prompt'
)

parser.add_argument(
    '--init_image', type=str, required=False, default=None, help='init image to use'
)

parser.add_argument(
    '--skip_timesteps',
    type=int,
    required=False,
    default=0,
    help='how many diffusion steps are gonna be skipped',
)

parser.add_argument(
    '--prefix', type=str, required=False, default='', help='prefix for output files'
)

parser.add_argument(
    '--output_path', type=str, required=False, default='', help='output dir'
)

parser.add_argument(
    '--num_batches', type=int, default=1, required=False, help='number of batches'
)

parser.add_argument(
    '--batch_size', type=int, default=1, required=False, help='batch size'
)

parser.add_argument(
    '--width',
    type=int,
    default=256,
    required=False,
    help='image size of output (multiple of 8)',
)

parser.add_argument(
    '--height',
    type=int,
    default=256,
    required=False,
    help='image size of output (multiple of 8)',
)

parser.add_argument('--seed', type=int, default=-1, required=False, help='random seed')

parser.add_argument(
    '--guidance_scale',
    type=float,
    default=5.0,
    required=False,
    help='classifier-free guidance scale',
)

parser.add_argument(
    '--steps', type=int, default=300, required=False, help='number of diffusion steps'
)

parser.add_argument('--cpu', dest='cpu', action='store_true')

parser.add_argument('--clip_guidance', dest='clip_guidance', action='store_true')

parser.add_argument(
    '--clip_guidance_scale',
    type=float,
    default=150,
    required=False,
    help='Controls how much the image should look like the prompt',
)  # may need to use lower value for ddim

parser.add_argument(
    '--cutn', type=int, default=16, required=False, help='Number of cuts'
)

parser.add_argument(
    '--ddim', dest='ddim', action='store_true'
)  # turn on to use 50 step ddim

parser.add_argument(
    '--ddpm', dest='ddpm', action='store_true'
)  # turn on to use 50 step ddim

static_args = parser.parse_args([])
