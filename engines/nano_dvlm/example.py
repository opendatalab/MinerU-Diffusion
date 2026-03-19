from pathlib import Path
import shutil
from nanovllm import LLM, SamplingParams
from termcolor import cprint

REPO_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_DIR / "model"

def build_message(image: Path):
    
    question = '\nPlease describe the image in detail:'
    
    message = [
        {"role": "user", "content": [
            {"type": "image", "image": str(image)},
            {"type": "text", "text": question},
        ]},
    ]
    return message

if __name__ == "__main__":
    block_size = 32
    llm = LLM(
        str(MODEL_PATH),
        enforce_eager=False,
        tensor_parallel_size=1,
        mask_token_id=151669,
        block_size=block_size,
        # max_num_seqs=32,
        max_model_len=8192,
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        max_new_tokens=4096,
        denoising_strategy="low_confidence_dynamic",
        dynamic_threshold=0.9,
        stop_tokens=['<|endoftext|>', '<|im_end|>'],
    )

    messages = []
    images = [str(REPO_DIR / "assets" / "image.png")]
    for image in images:
        messages.append(build_message(image))

    results = llm.generate_messages(messages, sampling_params=sampling_params)
    text = results[0]['text']
    text = text.split("<|im_end|>")[0].strip()
    cprint(text, 'green')
