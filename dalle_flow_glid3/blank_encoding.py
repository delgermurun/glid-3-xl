import numpy as np

from .model import bert, model_config, device

def generate_blank_embeddings(blank_text, clip_blank_encoding):
    blank_bert_embedding = (
        bert.encode([blank_text])
    ).to(device).float()

    blank_clip_embedding = np.array(clip_blank_encoding).astype(
        model_config['use_fp16'] and np.float16 or np.float32
    ).reshape(1, -1)

    return blank_bert_embedding, blank_clip_embedding