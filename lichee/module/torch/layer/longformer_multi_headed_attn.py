# -*- coding: utf-8 -*-
# _*_ conding:utf-8 _*_
# Author : Nick
# Time : 2020/9/15  3:21 下午

from typing import List, Tuple

import torch
import math


def nonzero_tuple(x):
    if x.dim() == 0:
        return x.unsqueeze(0).nonzero().unbind(1)
    return x.nonzero().unbind(1)


class LongformerSelfAttention(torch.nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__()
        if cfg["CONFIG"]["HIDDEN_SIZE"] % cfg["CONFIG"]["NUM_ATTENTION_HEADS"] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cfg["CONFIG"]["HIDDEN_SIZE"], cfg["CONFIG"]["NUM_ATTENTION_HEADS"])
            )
        self.num_heads = cfg["CONFIG"]["NUM_ATTENTION_HEADS"]
        self.head_dim = int(cfg["CONFIG"]["HIDDEN_SIZE"] / cfg["CONFIG"]["NUM_ATTENTION_HEADS"])
        self.embed_dim = cfg["CONFIG"]["HIDDEN_SIZE"]

        self.query = torch.nn.Linear(cfg["CONFIG"]["HIDDEN_SIZE"], self.embed_dim)
        self.key = torch.nn.Linear(cfg["CONFIG"]["HIDDEN_SIZE"], self.embed_dim)
        self.value = torch.nn.Linear(cfg["CONFIG"]["HIDDEN_SIZE"], self.embed_dim)

        # separate projection layers for tokens with global attention
        self.query_global = torch.nn.Linear(cfg["CONFIG"]["HIDDEN_SIZE"], self.embed_dim)
        self.key_global = torch.nn.Linear(cfg["CONFIG"]["HIDDEN_SIZE"], self.embed_dim)
        self.value_global = torch.nn.Linear(cfg["CONFIG"]["HIDDEN_SIZE"], self.embed_dim)

        self.dropout = cfg["CONFIG"]["ATTENTION_PROBS_DROPOUT_PROB"]

        self.layer_id = layer_id  # 待补充超参数
        attention_window = cfg["CONFIG"]["ATTENTION_WINDOW"][self.layer_id]  # 待补充超参数
        assert (
                attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
                attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

    def forward(
            self, hidden_states, attention_mask
    ):
        """
        LongformerSelfAttention expects `len(hidden_states)` to be multiple of `attention_window`.
        Padding to `attention_window` happens in LongformerModel.forward to avoid redoing the padding on each layer.

        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention

        """
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)

        # is index masked or global attention
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        hidden_states = hidden_states.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
                embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # attn_probs = (batch_size, seq_len, num_heads, window*2+1)
        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, -10000.0
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            torch.ones(size=float_mask.size(), dtype=float_mask.dtype, device=float_mask.device),
            float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], f"attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, " \
           f"{self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"

        max_num_global_attn_indices = torch.tensor(0)
        is_index_global_attn_nonzero = [torch.tensor(0)]
        is_local_index_global_attn_nonzero = [torch.tensor(0)]
        is_local_index_no_global_attn_nonzero = [torch.tensor(0)]

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            ret = self._get_global_attn_indices(is_index_global_attn)
            max_num_global_attn_indices = ret[0]
            is_index_global_attn_nonzero = ret[1]
            is_local_index_global_attn_nonzero = ret[2]
            is_local_index_no_global_attn_nonzero = ret[3]
            # calculate global attn probs from global key

            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

            # free memory
            # if self.query.training:
            #     del global_key_attn_scores


        attn_probs_fp32 = torch.nn.functional.softmax(attn_scores, dim=-1,
                                                      dtype=torch.float32)  # use fp32 for numerical stability
        attn_probs = attn_probs_fp32.type_as(attn_scores)

        # free memory
        # if self.query.training:
        #     del attn_probs_fp32

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)

        # apply dropout
        attn_probs = torch.nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            global_attn_output = self._compute_global_attn_output(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                                         is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
                                         ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )

        attn_output = attn_output.transpose(0, 1)

        return attn_output

    def _pad_and_transpose_last_two_dims(self, hidden_states_padded, padding: Tuple[int, int, int, int]):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = torch.nn.functional.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            hidden_states_padded.size(0), hidden_states_padded.size(1), hidden_states_padded.size(3),
            hidden_states_padded.size(2)
        )
        return hidden_states_padded

    def _pad_and_diagonalize(self, chunked_hidden_states):
        """shift every row 1 step right, converting columns into diagonals.
           Example:
                 chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                          -1.8348,  0.7672,  0.2986,  0.0285,
                                          -0.7584,  0.4206, -0.0405,  0.1599,
                                          2.0514, -1.1600,  0.5372,  0.2629 ]
                 window_overlap = num_rows = 4
                (pad & diagonilize) =>
                [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
                  0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
                  0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
                  0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        chunked_hidden_states = torch.nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        # Padding value is not important because it'll be overwritten
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
                                :, :, :-window_overlap
                                ]  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )  # total_num_heads x num_chunks, window_overlap x hidden_dim+window_overlap
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    def _chunk(self, hidden_states, window_overlap: int):
        """convert into overlapping chunkings. Chunk size = 2w, overlap size = w"""

        # non-overlapping chunks of size = 2w
        hidden_states = hidden_states.view(
            hidden_states.size(0),
            hidden_states.size(1) // (window_overlap * 2),
            window_overlap * 2,
            hidden_states.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(hidden_states.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = [hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
                        hidden_states.stride(3)]
        chunk_stride[1] = chunk_stride[1] // 2
        return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

    def _mask_invalid_locations(self, input_tensor, affected_seq_len: int):
        beginning_mask_2d = torch.ones(affected_seq_len, affected_seq_len + 1, dtype=input_tensor.dtype,
                                       device=input_tensor.device).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """Matrix multiplication of query and key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
        with an overlap of size window_overlap"""
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
                seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        chunked_query = self._chunk(query, window_overlap)
        chunked_key = self._chunk(key, window_overlap)

        # matrix multipication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x window_overlap
        chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (chunked_query, chunked_key))  # multiply

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower
        # triangles (attention from a word to window_overlap previous words). The following column is attention
        # score from each word to itself, then followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                                :, :, :window_overlap, : window_overlap + 1
                                                                ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                               :, -1, window_overlap:, : window_overlap + 1
                                                               ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
                                                               :, :, -(window_overlap + 1): -1, window_overlap + 1:
                                                               ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
                                                                              :, 0, : window_overlap - 1,
                                                                              1 - window_overlap:
                                                                              ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn(
            self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors.
           Returned tensor will be of the same shape as `attn_probs`"""
        batch_size, seq_len, num_heads, head_dim = value.size()

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # group batch_size and num_heads dimensions into one
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = torch.nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1.0)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = [padded_value.stride(0), padded_value.stride(1), padded_value.stride(2)]
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

    def _get_global_attn_indices(self, is_index_global_attn):
        """ compute global attn indices required throughout forward pass """
        # helper variable
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # indices of global attn
        is_index_global_attn_nonzero = nonzero_tuple(is_index_global_attn)

        # helper variable
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = nonzero_tuple(is_local_index_global_attn)

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = nonzero_tuple(is_local_index_global_attn == 0)
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
            self,
            key_vectors,
            query_vectors,
            max_num_global_attn_indices,
            is_index_global_attn_nonzero: List[torch.Tensor],
            is_local_index_global_attn_nonzero: List[torch.Tensor],
            is_local_index_no_global_attn_nonzero: List[torch.Tensor],
    ):
        batch_size = key_vectors.shape[0]

        # create only global key vectors
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )

        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query_vectors, key_vectors_only_global))

        attn_probs_from_global_key[
        is_local_index_no_global_attn_nonzero[0], :, :, is_local_index_no_global_attn_nonzero[1]
        ] = -10000.0

        return attn_probs_from_global_key

    def _compute_attn_output(
            self,
            value_vectors,
            attn_probs,
            max_num_global_attn_indices,
            is_index_global_attn_nonzero: List[torch.Tensor],
            is_local_index_global_attn_nonzero: List[torch.Tensor],
    ):
        batch_size = attn_probs.shape[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        # get value vectors for global only
        value_vectors_only_global = value_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]

        # use `matmul` because `einsum` crashes sometimes with fp16
        # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
        # compute attn output only global
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.transpose(1, 2), value_vectors_only_global.transpose(1, 2)
        ).transpose(1, 2)

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output(
            self,
            hidden_states,
            max_num_global_attn_indices,
            is_local_index_global_attn_nonzero: List[torch.Tensor],
            is_index_global_attn_nonzero: List[torch.Tensor],
            is_local_index_no_global_attn_nonzero: List[torch.Tensor],
            is_index_masked,
    ):
        seq_len, batch_size = hidden_states.shape[:2]

        # prepare global hidden states
        global_attn_hidden_states = hidden_states.new_zeros(max_num_global_attn_indices, batch_size, self.embed_dim)
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = hidden_states[
            is_index_global_attn_nonzero[::-1]
        ]

        # global key, query, value
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)

        # normalize
        global_query_vectors_only_global /= math.sqrt(self.head_dim)

        # reshape
        global_query_vectors_only_global = (
            global_query_vectors_only_global.contiguous()
                .view(max_num_global_attn_indices, batch_size * self.num_heads, self.head_dim)
                .transpose(0, 1)
        )  # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
        global_key_vectors = (
            global_key_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)
        global_value_vectors = (
            global_value_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)

        # compute attn scores
        global_attn_scores = torch.bmm(global_query_vectors_only_global, global_key_vectors.transpose(1, 2))

        assert global_attn_scores.size(0) == batch_size * self.num_heads, \
            f"global_attn_scores have the wrong size. size(0) should be {batch_size * self.num_heads}, " \
            f"but is {global_attn_scores.size(0)}."
        assert global_attn_scores.size(1) == max_num_global_attn_indices, \
            f"global_attn_scores have the wrong size. size(1) should be {max_num_global_attn_indices}, " \
            f"but is {global_attn_scores.size(1)}."
        assert global_attn_scores.size(2) == seq_len, \
            f"global_attn_scores have the wrong size. size(2) should be {seq_len}, but is {global_attn_scores.size(2)}."

        global_attn_scores = global_attn_scores.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)

        global_attn_scores[
        is_local_index_no_global_attn_nonzero[0], :, is_local_index_no_global_attn_nonzero[1], :
        ] = -10000.0

        global_attn_scores = global_attn_scores.masked_fill(is_index_masked[:, None, None, :], -10000.0, )

        global_attn_scores = global_attn_scores.view(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)

        # compute global attn probs
        global_attn_probs_float = torch.nn.functional.softmax(
            global_attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        global_attn_probs = torch.nn.functional.dropout(
            global_attn_probs_float.type_as(global_attn_scores), p=self.dropout, training=self.training
        )

        # global attn output
        global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)

        assert global_attn_output.size(0) == batch_size * self.num_heads, \
            f"global_attn_scores have the wrong size. size(0) should be {batch_size * self.num_heads}, " \
            f"but is {global_attn_output.size(0)}."
        assert global_attn_output.size(1) == max_num_global_attn_indices, \
            f"global_attn_scores have the wrong size. size(1) should be {max_num_global_attn_indices}, " \
            f"but is {global_attn_output.size(1)}."
        assert global_attn_output.size(2) == self.head_dim, \
            f"global_attn_scores have the wrong size. size(2) should be {self.head_dim}, " \
            f"but is {global_attn_output.size(2)}."

        global_attn_output = global_attn_output.view(
            batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim
        )
        return global_attn_output
