import torch
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    CACHING = auto()
    FINISHED = auto()


class Sequence:
    kvcache_block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params = SamplingParams(),
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        denoising_block_size: int | None = None,
        mask_token_id: int | None = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.step_map = []
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids) # generated tokens, no mask token
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.sampling_params = sampling_params
        self.temperature = sampling_params.temperature
        self.max_new_tokens = sampling_params.max_new_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.pixel_values = pixel_values
        self.image_grid_thw = image_grid_thw
        self.denoising_block_size = denoising_block_size
        self.mask_token_id = mask_token_id
        self.current_denoising_block = torch.full((self.denoising_block_size,), self.mask_token_id, dtype=torch.int64, device='cpu')
        self.current_step_map = torch.empty((self.denoising_block_size,), dtype=torch.int32, device='cpu')
        self.current_denoise_step = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED
    
    @property
    def is_caching(self):
        return self.status == SequenceStatus.CACHING

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.kvcache_block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.kvcache_block_size - 1) // self.kvcache_block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.kvcache_block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.kvcache_block_size: (i+1)*self.kvcache_block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def append_block(self):
        self.token_ids.extend(self.current_denoising_block.tolist())
        self.step_map.extend(self.current_step_map.tolist())
        self.num_tokens += self.denoising_block_size
        self.current_denoising_block = torch.full((self.denoising_block_size,), self.mask_token_id, dtype=torch.int64, device='cpu')

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]

    def need_new_block(self):
        return self.num_tokens + self.denoising_block_size > self.kvcache_block_size * len(self.block_table)

    def get_slots(self, lo: int, hi: int):
        if not self.block_table:
            return [-1] * (hi - lo)
        slots = []
        lo_block = lo // self.kvcache_block_size
        hi_block = hi // self.kvcache_block_size
        if lo_block == hi_block:
            start = self.block_table[lo_block] * self.kvcache_block_size + (lo % self.kvcache_block_size)
            slots.extend(list(range(start, start + (hi - lo))))
            return slots
        slots.extend(list(range(
            self.block_table[lo_block] * self.kvcache_block_size + (lo % self.kvcache_block_size),
            self.block_table[lo_block] * self.kvcache_block_size + self.kvcache_block_size,
        )))
        for block_idx in range(lo_block + 1, hi_block):
            slots.extend(list(range(
                self.block_table[block_idx] * self.kvcache_block_size,
                self.block_table[block_idx] * self.kvcache_block_size + self.kvcache_block_size,
            )))
        if hi % self.kvcache_block_size > 0:
            slots.extend(list(range(
                self.block_table[hi_block] * self.kvcache_block_size,
                self.block_table[hi_block] * self.kvcache_block_size + (hi % self.kvcache_block_size),
            )))
        return slots
