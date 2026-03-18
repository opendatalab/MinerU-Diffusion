import torch
from torch import nn
from nanovllm.engine.sequence import Sequence


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def prepare(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float() / temperatures.unsqueeze(1)

        gumbel = -torch.empty_like(logits).exponential_().log()
        gumbel_logits = logits + gumbel

        x0 = gumbel_logits.argmax(dim=-1)  # [bs, seq_len]

        probs = torch.softmax(logits, dim=-1)
        x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        return x0, x0_p

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, seqs: list[Sequence]):
        x0, x0_p = self.prepare(logits, temperatures)
        x0, x0_p = x0.cpu(), x0_p.cpu()
        
        x = []
        for seq in seqs:
            x.append(seq.current_denoising_block)
        x = torch.stack(x, dim=0)  # [bs, seq_len]
        mask_index = (x == seqs[0].mask_token_id)
        
        # Only support same denoising strategy for a batch
        denoising_strategy = seqs[0].sampling_params.denoising_strategy
        if denoising_strategy == 'low_confidence_dynamic':
            dynamic_threshold = seqs[0].sampling_params.dynamic_threshold
            # import ipdb; ipdb.set_trace()
            confidence = torch.where(mask_index, x0_p, -torch.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                if not mask_index[j].any():
                    continue
                high_conf_mask = confidence[j] > dynamic_threshold
                num_high_confidence = high_conf_mask.sum()
                if num_high_confidence >= 1: # todo: support custom denoising step
                    transfer_index[j] = high_conf_mask
                else:
                    _, idx = torch.topk(confidence[j], 1)
                    transfer_index[j, idx] = True
        else:
            raise NotImplementedError(f"Denoising strategy {denoising_strategy} not implemented")
        
        x[transfer_index] = x0[transfer_index]
        
        for i, seq in enumerate(seqs):
            seq.current_denoising_block = x[i]
            seq.current_step_map[transfer_index[i]] = seq.current_denoise_step
            seq.current_denoise_step += 1

        return transfer_index.sum().item()
