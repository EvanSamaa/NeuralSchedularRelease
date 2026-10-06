"""The scheduler language model.

GLM-4-Voice (ChatGLM) with two additions:

* a 3-way *channel embedding* (script / dialogue-A / dialogue-B) added to the
  queries and keys of every attention layer after the rotary embedding;
* an optional *duration head* that predicts, for every position, how many more
  frames the token at that position repeats in its own channel.

The input sequence is::

    <|start-of-script-A|> script A <|end-of-script-A|>
    <|start-of-script-B|> script B <|end-of-script-B|>
    <|start-of-dialogue|> A0 B0 A1 B1 ... <|end-of-dialogue|>

``generation_start_index`` is the index of ``A0``. Script tokens get one rotary
position each; the two dialogue tokens of a frame share a position.

The transformer blocks derive from the GLM-4-Voice implementation
(https://github.com/THUDM/GLM-4-Voice, Apache-2.0).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn.utils import skip_init
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration import ChatGLMConfig

try:
    from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

    if is_flash_attn_2_available():
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    pass

logger = logging.get_logger(__name__)


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


def compute_channel_ids(
    seq_length: int,
    generation_start_index: Union[int, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Channel id of every position, derived from where the dialogue starts.

    Args:
        seq_length: sequence length.
        generation_start_index: ``[batch]`` tensor with the index of the first
            dialogue token of each sample, or a single int.
        device: device of the result.
        dtype: dtype of the result.

    Returns:
        ``[batch, seq_length]`` tensor: 0 before ``generation_start_index``
        (script), then alternating 1 (dialogue A) and 2 (dialogue B).
    """
    if isinstance(generation_start_index, int):
        generation_start_index = torch.tensor([generation_start_index], device=device)

    batch_size = generation_start_index.shape[0]
    positions = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
    gen_start = generation_start_index.unsqueeze(1)
    relative_pos = positions - gen_start

    channel_ids = torch.zeros(batch_size, seq_length, device=device, dtype=dtype)
    channel_ids = torch.where(positions >= gen_start, (relative_pos % 2) + 1, channel_ids)
    return channel_ids.to(dtype)


@dataclass
class CausalLMOutputWithPastAndDuration(ModelOutput):
    """Causal LM output extended with the duration head's logits and losses."""

    loss: Optional[torch.FloatTensor] = None
    token_loss: Optional[torch.FloatTensor] = None
    duration_loss: Optional[torch.FloatTensor] = None
    duration_accuracy: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    duration_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        base = base * self.rope_ratio
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


class ChatGLMChannelEmbedding(nn.Module):
    """Learned embedding of the channel id, one vector of size ``kv_channels`` per channel.

    The table is created in half precision (bfloat16 when the GPU supports it,
    float16 otherwise) and is cast with the rest of the model afterwards.
    """

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.embedding_size = config.kv_channels
        self.num_channels = config.num_channel
        emb_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.channel_embedding = nn.Embedding(self.num_channels, self.embedding_size, dtype=emb_dtype)

    def forward(self, x):
        return self.channel_embedding(x)


def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1).type_as(x)


def apply_chatglm_channel_emb(q, k, channel_emb_module, device, channel_ids=None):
    """Add the channel embedding to the query and key tensors.

    Args:
        q: ``[batch, q_heads, q_len, head_dim]``.
        k: ``[batch, k_heads, k_len, head_dim]`` (the head counts differ under
            multi-query attention).
        channel_emb_module: the :class:`ChatGLMChannelEmbedding`.
        device: device for the fallback indices.
        channel_ids: ``[batch, seq_len]``. The last ``q_len`` / ``k_len``
            entries are used, so the new tokens of a cached decoding step line
            up. Without it, channels alternate 0/1 by position.
    """
    bsz_q, num_q_heads, q_len, _ = q.shape
    bsz_k, num_k_heads, k_len, _ = k.shape

    if channel_ids is not None:
        q_channel_idx = channel_ids[:, -q_len:].unsqueeze(1).expand(bsz_q, num_q_heads, q_len)
        k_channel_idx = channel_ids[:, -k_len:].unsqueeze(1).expand(bsz_k, num_k_heads, k_len)
    else:
        q_channel_idx = (torch.arange(q_len, device=device) % 2).expand(bsz_q, num_q_heads, q_len)
        k_channel_idx = (torch.arange(k_len, device=device) % 2).expand(bsz_k, num_k_heads, k_len)

    q_embed = q + channel_emb_module(q_channel_idx)
    k_embed = k + channel_emb_module(k_channel_idx)
    return q_embed.to(q.dtype), k_embed.to(q.dtype)


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class CoreAttention(torch.nn.Module):
    def __init__(self, config: ChatGLMConfig, layer_number):
        super(CoreAttention, self).__init__()
        self.config = config
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.is_causal = True

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # [b, np, sq, sk]
        output_size = (query_layer.size(0), query_layer.size(1), query_layer.size(2), key_layer.size(2))

        # [b, np, sq, hn] -> [b * np, sq, hn]
        query_layer = query_layer.view(output_size[0] * output_size[1], output_size[2], -1)
        # [b, np, sk, hn] -> [b * np, sk, hn]
        key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = torch.empty(
            output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
            device=query_layer.device
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,  # [b * np, sq, hn]
            key_layer.transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                        device=attention_scores.device, dtype=torch.bool)
            attention_mask.tril_()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # query layer shape: [b * np, sq, hn]
        # value layer shape: [b, np, sk, hn]
        # attention shape: [b, np, sq, sk]
        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(0), value_layer.size(1), query_layer.size(1), value_layer.size(3))
        # change view [b * np, sk, hn]
        value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(2), -1)
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [b, sq, np, hn]
        context_layer = context_layer.transpose(1, 2).contiguous()
        # [b, sq, np, hn] --> [b, sq, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)

        return context_layer


class SdpaAttention(CoreAttention):
    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                             is_causal=True,
                                                                             dropout_p=self.config.attention_dropout if self.training else 0.0)
        else:
            if attention_mask is not None:
                attention_mask = ~attention_mask
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                             attention_mask,
                                                                             dropout_p=self.config.attention_dropout if self.training else 0.0)
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2
class FlashAttention2(CoreAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(self, query_states, key_states, value_states, attention_mask):
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        batch_size, query_length = query_states.shape[:2]
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1
        dropout = self.config.attention_dropout if self.training else 0.0
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=None,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=None, causal=causal
            )
        attn_output = attn_output.reshape(batch_size, query_length, self.hidden_size_per_partition).contiguous()
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads_per_partition, head_dim),
                indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


CORE_ATTENTION_CLASSES = {
    "eager": CoreAttention,
    "sdpa": SdpaAttention,
    "flash_attention_2": FlashAttention2
}

class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias,
                                         device=device, **_config_to_kwargs(config)
                                         )

        self.core_attention = CORE_ATTENTION_CLASSES[config._attn_implementation](config, self.layer_number)

        # Output.
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                               device=device, **_config_to_kwargs(config)
                               )

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, channel_embeddings, kv_cache=None, use_cache=True,
            channel_ids=None,
    ):
        # hidden_states: [b, sq, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [b, sq, h] --> [b, sq, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        if channel_embeddings is not None:
            query_layer, key_layer = apply_chatglm_channel_emb(
                query_layer, key_layer, channel_embeddings, query_layer.device, channel_ids=channel_ids
            )

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=2)
            value_layer = torch.cat((cache_v, value_layer), dim=2)
        if use_cache:
            if kv_cache is None:
                kv_cache = torch.cat((key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)),
                                     dim=1)
            else:
                kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(2)
            key_layer = key_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
            )
            value_layer = value_layer.unsqueeze(2)
            value_layer = value_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
            )

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMConfig, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                             dtype=config.torch_dtype)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                      dtype=config.torch_dtype)

        # MLP
        self.mlp = MLP(config, device=device)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, channel_embeddings, kv_cache=None, use_cache=True,
            channel_ids=None,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            channel_embeddings,
            kv_cache=kv_cache,
            use_cache=use_cache,
            channel_ids=channel_ids,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache


class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                 dtype=config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, channel_embeddings, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
            channel_ids: Optional[torch.Tensor] = None,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    channel_embeddings,
                    kv_caches[index],
                    use_cache,
                    use_reentrant=False,
                    channel_ids=channel_ids,
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    channel_embeddings,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache,
                    channel_ids=channel_ids,
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                # token by token decoding, use tuple format
                if kv_caches[0] is not None:
                    presents = presents + (kv_cache,)
                # prefilling in decoding, use tensor format to save cuda memory
                else:
                    if len(presents) == 0:
                        presents = kv_cache
                    else:
                        presents = torch.cat((presents, kv_cache.to(presents.device)), dim=0)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMPreTrainedModel(PreTrainedModel):
    """Weight handling and the loading interface shared by the scheduler models."""

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module):
        """Modules keep the initialisation done in their constructors."""
        return

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        """Boolean attention mask for the eager and SDPA attention paths.

        Returns ``[batch, 1, seq_len, past_len + seq_len]`` with ``True`` where
        attention is *not* allowed (future positions and padding).
        """
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if past_length:
            full_attention_mask = torch.cat(
                (torch.ones(batch_size, seq_length, past_length, device=input_ids.device), full_attention_mask),
                dim=-1,
            )
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def get_position_ids(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids


class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
            device=device
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings

class DurationPredictionHead(nn.Module):
    """Two-layer MLP from hidden states to ``num_duration_classes`` logits.

    Class ``d`` means the token at this position repeats ``d`` more frames in
    its channel; the last class means "that many or more". The final layer is
    initialised small (std 0.02, zero bias) so the logits start near zero.
    """

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_duration_classes = getattr(config, "num_duration_classes", 32)

        self.duration_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 512, dtype=config.torch_dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_duration_classes, dtype=config.torch_dtype),
        )
        nn.init.normal_(self.duration_predictor[-1].weight, std=0.02)
        nn.init.zeros_(self.duration_predictor[-1].bias)

    def forward(self, hidden_states):
        # Logits are returned in float32 for a stable cross-entropy.
        return self.duration_predictor(hidden_states).float()


class ChatGLMModel(ChatGLMPreTrainedModel):
    """Token embedding, channel embedding, rotary positions, transformer and output layer."""

    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.embedding = init_method(Embedding, config, **init_kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.channel_emb = ChatGLMChannelEmbedding(config)
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, rope_ratio=config.rope_ratio,
                                              original_impl=config.original_rope,
                                              device=device, dtype=config.torch_dtype)
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        self.output_layer = init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
                                        dtype=config.torch_dtype, **init_kwargs)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    def forward(
        self,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        channel_ids: Optional[torch.Tensor] = None,
    ):
        if channel_ids is None:
            raise ValueError("channel_ids is required; pass it or give generation_start_index to the LM wrapper")
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.config._attn_implementation == "flash_attention_2":
            # Flash attention is causal by itself and only needs the 2-D padding mask.
            if attention_mask is not None and not attention_mask.all():
                encoder_attention_mask = attention_mask
            else:
                encoder_attention_mask = None
        else:
            if full_attention_mask is None:
                if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                    full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
            encoder_attention_mask = full_attention_mask

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            position_ids = position_ids.repeat(batch_size, 1)

        # Rotary positions: script tokens count 0, 1, 2, ...; the two dialogue
        # tokens of a frame share one position, continuing after the script.
        # Every position with channel id 0 counts as script here, and that
        # includes right-padding when channel ids were derived from
        # generation_start_index (see docs/model.md, "Known issues").
        is_script = (channel_ids == 0)
        script_positions = is_script.long().cumsum(dim=-1) - 1
        script_positions = script_positions.clamp(min=0)

        num_script_tokens = is_script.long().sum(dim=-1, keepdim=True)
        dialogue_offset = (position_ids - num_script_tokens).clamp(min=0)
        dialogue_positions = num_script_tokens + (dialogue_offset // 2)
        dual_channel_position_ids = torch.where(is_script, script_positions, dialogue_positions)

        max_pos_needed = dual_channel_position_ids.max().item()
        rotary_pos_emb_cache = self.rotary_pos_emb(max_pos_needed + 1)
        rotary_pos_emb = rotary_pos_emb_cache[dual_channel_position_ids]

        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, encoder_attention_mask, rotary_pos_emb=rotary_pos_emb, channel_embeddings=self.channel_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states,
            channel_ids=channel_ids,
        )
        if presents is not None and type(presents) is torch.Tensor:
            presents = presents.split(1, dim=0)
            presents = list(presents)
            presents = [list(x.squeeze(0).split(1, dim=0)) for x in presents]
            presents = [tuple([x.squeeze(0) for x in y]) for y in presents]
            presents = tuple(presents)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class _SchedulerLM(ChatGLMPreTrainedModel):
    """Shared implementation of the two scheduler LM classes."""

    has_duration_head = False

    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)
        self.config = config
        if self.has_duration_head:
            self.duration_head = DurationPredictionHead(config)

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "The scheduler is driven by explicit forward passes (it needs channel_ids and "
            "generation_start_index on every step); Hugging Face generate() is not supported."
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        channel_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
        generation_start_index: Optional[torch.Tensor] = None,
        remaining_durations: Optional[torch.Tensor] = None,
        duration_loss_weight: float = 0.5,
    ):
        """Run the LM.

        Args:
            input_ids: ``[batch, seq]`` vocab ids.
            channel_ids: ``[batch, seq]`` in {0, 1, 2}. Computed from
                ``generation_start_index`` when omitted, with padded positions
                set to 0.
            attention_mask: ``[batch, seq]`` padding mask; all ones if omitted.
            labels: ``[batch, seq]`` targets for next-token cross-entropy
                (-100 is ignored; the script region is normally masked).
            generation_start_index: ``[batch]`` index of the first dialogue token.
            remaining_durations: ``[batch, seq]`` duration labels (-100 is
                ignored), aligned with ``input_ids`` and not shifted. Values
                above the last class are clamped to it. Duration model only.
            duration_loss_weight: weight of the duration loss in ``loss``.
        """
        if remaining_durations is not None and not self.has_duration_head:
            raise ValueError("remaining_durations was given but this model has no duration head")
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if channel_ids is None and generation_start_index is not None and input_ids is not None:
            batch_size, seq_length = input_ids.shape
            channel_ids = compute_channel_ids(
                seq_length=seq_length,
                generation_start_index=generation_start_index,
                device=input_ids.device,
            )
            channel_ids = channel_ids * attention_mask.to(channel_ids.dtype)

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            channel_ids=channel_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_attentions=output_attentions,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[:, -1:]
        lm_logits = self.transformer.output_layer(hidden_states)

        duration_logits = self.duration_head(hidden_states) if self.has_duration_head else None

        loss = None
        token_loss = None
        duration_loss = None
        duration_accuracy = None

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=-100)
            token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            if remaining_durations is not None:
                duration_labels = remaining_durations.contiguous()
                max_class = self.duration_head.num_duration_classes - 1
                duration_labels_clipped = torch.where(
                    duration_labels == -100,
                    duration_labels,
                    torch.clamp(duration_labels, min=0, max=max_class),
                )

                duration_loss_fct = CrossEntropyLoss(ignore_index=-100)
                duration_loss = duration_loss_fct(
                    duration_logits.view(-1, duration_logits.size(-1)),
                    duration_labels_clipped.view(-1),
                )

                with torch.no_grad():
                    duration_preds = duration_logits.argmax(dim=-1)
                    valid_mask = duration_labels_clipped != -100
                    if valid_mask.sum() > 0:
                        exact_match = (duration_preds == duration_labels_clipped) & valid_mask
                        duration_accuracy = exact_match.sum().float() / valid_mask.sum().float()
                    else:
                        duration_accuracy = torch.tensor(0.0, device=duration_logits.device)

                loss = token_loss + duration_loss_weight * duration_loss
            else:
                loss = token_loss
                if self.has_duration_head:
                    duration_loss = torch.tensor(0.0, device=lm_logits.device)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Loss is {loss} (token loss {token_loss}, duration loss {duration_loss})")

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        if not self.has_duration_head:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )

        return CausalLMOutputWithPastAndDuration(
            loss=loss,
            token_loss=token_loss,
            duration_loss=duration_loss,
            duration_accuracy=duration_accuracy,
            logits=lm_logits,
            duration_logits=duration_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class ChatGLMForConditionalGeneration(_SchedulerLM):
    """Scheduler LM without the duration head (``use_duration_prediction=False``)."""

    has_duration_head = False


class ChatGLMForConditionalGenerationPredDuration(_SchedulerLM):
    """Scheduler LM with the duration head. This is the class of the released checkpoint."""

    has_duration_head = True


def scheduler_model_class(config: ChatGLMConfig):
    """The LM class a config asks for."""
    if getattr(config, "use_duration_prediction", False):
        return ChatGLMForConditionalGenerationPredDuration
    return ChatGLMForConditionalGeneration
