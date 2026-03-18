import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.processors.processor import Processor


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(config, 0, self.events)
        
        model_type = config.hf_config.model_type
        if model_type in ["dmllm", "mineru_diffusion"]:
            self.processor = Processor(model, config.hf_config.image_token_id)
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        self.config = config
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def add_request_message(self, message: list[dict], sampling_params: SamplingParams):
        prompt, images = self.processor.apply_chat_template(message)
        inputs = self.processor.process(prompt, images)
        sampling_params.stop_token_ids = self.tokenizer.convert_tokens_to_ids(sampling_params.stop_tokens)
        seq = Sequence(**inputs, sampling_params=sampling_params, denoising_block_size=self.config.block_size, mask_token_id=self.config.mask_token_id)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # import ipdb; ipdb.set_trace()
        num_tokens = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.step_map) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) + num_tokens if is_prefill else -num_tokens
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate_routine(self, use_tqdm: bool = True, pbar = None):
        outputs = {}
        prefill_throughput = decode_throughput = total_throughput = 0.
        start = perf_counter()
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                    total_throughput += num_tokens
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                    total_throughput += -num_tokens
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids, step_map in output:
                outputs[seq_id] = (token_ids, step_map)
                if use_tqdm:
                    pbar.update(1)
        print(f"Average Throughput: {total_throughput / (perf_counter() - start):.3f} tok/s")
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids, "step_map": step_map} for token_ids, step_map in outputs]
        if pbar:
            pbar.close()
        return outputs

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = None
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        return self.generate_routine(use_tqdm, pbar)

    def generate_messages(
        self,
        messages: list[dict] | list[list[dict]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ):
        pbar = None
        if use_tqdm:
            pbar = tqdm(total=len(messages), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(messages)
        for messsage, sp in zip(messages, sampling_params): # image preprocessing inside
            self.add_request_message(messsage, sp)
        return self.generate_routine(use_tqdm, pbar)
