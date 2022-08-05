import json
import requests
import numpy as np

from .model import bert, model_config, device

def generate_blank_embeddings(blank, clip_as_service_url):
    blank_bert_embedding = (
        bert.encode([blank])
    ).to(device).float()

    res = requests.post(
        f'{clip_as_service_url}/post',
        data=json.dumps({'execEndpoint':'/', 'data': [{'text': blank}]}),
        headers={'content-type': 'application/json'}
    ).json()

    blank_clip_embedding = np.array(res['data'][0]['embedding']).astype(
        model_config['use_fp16'] and np.float16 or np.float32
    ).reshape(1, -1)

    return blank_bert_embedding, blank_clip_embedding