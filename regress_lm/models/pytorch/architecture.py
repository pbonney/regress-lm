# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default PyTorch architecture for a RegressLM."""

import math
from regress_lm import vocabs
import torch
from torch import nn


class PositionalEncoding(nn.Module):
  """Default positional encoding."""

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.dropout(x + self.pe[:, : x.size(1)])


class EncoderDecoder(nn.Module):
  """Encoder-Decoder model in PyTorch."""

  def __init__(
      self,
      encoder_vocab: vocabs.EncoderVocab,
      decoder_vocab: vocabs.DecoderVocab,
      max_encoder_len: int,
      d_model: int,
      nhead: int,
      num_encoder_layers: int,
      num_decoder_layers: int,
      dim_feedforward: int,
      dropout: float,
  ):
    super().__init__()
    self.d_model = d_model
    self.encoder_pad_idx = encoder_vocab.pad_id
    self.src_tok_emb = nn.Embedding(len(encoder_vocab), d_model)
    self.tgt_tok_emb = nn.Embedding(len(decoder_vocab), d_model)
    self.encoder_positional_encoding = PositionalEncoding(
        d_model, dropout=dropout, max_len=max_encoder_len
    )
    self.decoder_positional_encoding = PositionalEncoding(
        d_model, dropout=dropout, max_len=decoder_vocab.decode_len + 1
    )
    self.transformer = nn.Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
    )
    self.generator = nn.Linear(d_model, len(decoder_vocab))

  def _generate_causal_mask(self, sz: int) -> torch.Tensor:
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

  def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
    """For training or calculating log_prob for the whole sequence.

    Args:
      src: (B, L_source) token ids.
      tgt_input: (B, L_target) token ids.

    Returns:
      (B, L_target, D_model) decoder output (logits) for the whole target.
    """
    src_padding_mask = src == self.encoder_pad_idx  # (Batch, L_source)
    tgt_causal_mask = self._generate_causal_mask(tgt_input.size(1))

    src_emb = self.encoder_positional_encoding(
        self.src_tok_emb(src) * math.sqrt(self.d_model)
    )
    tgt_emb = self.decoder_positional_encoding(
        self.tgt_tok_emb(tgt_input) * math.sqrt(self.d_model)
    )

    # Runs src_emb through its encoder, then runs tgt_emb and the encoder's
    # output (memory) through its decoder.
    transformer_output = self.transformer(
        src=src_emb,
        tgt=tgt_emb,
        tgt_mask=tgt_causal_mask,  # For decoder self-attention.
        src_key_padding_mask=src_padding_mask,  # Ignore source padding.
        tgt_key_padding_mask=None,  # No padding in tgt.
        memory_key_padding_mask=src_padding_mask,  # Ignore encoder padding.
    )
    # transformer_output: (B, L_target, D_model)
    return self.generator(transformer_output)

  def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes the source sequence.

    Args:
      src: (B, L_source)

    Returns:
      The encoder output (memory) and the source padding mask (for decoder).
    """
    src_padding_mask = src == self.encoder_pad_idx  # (B, L_source)
    src_emb = self.encoder_positional_encoding(
        self.src_tok_emb(src) * math.sqrt(self.d_model)
    )
    memory = self.transformer.encoder(
        src_emb, src_key_padding_mask=src_padding_mask
    )
    return memory, src_padding_mask  # memory: (B, L_source, D_model)

  def next_token_logits(
      self,
      current_tgt_seq: torch.Tensor,  # (B, current_tgt_len)
      memory: torch.Tensor,  # (B, L_source, D_model)
      memory_key_padding_mask: torch.Tensor,  # (B, L_source)
  ) -> torch.Tensor:
    """Decodes one step given the current target sequence and encoder memory.

    Args:
      current_tgt_seq: (B, current_tgt_len) - e.g., <start> t1, t2
      memory: (B, L_source, D_model) - Output from the encoder
      memory_key_padding_mask: (B, L_source) - Padding mask for the memory

    Returns:
      (B, V) - The logits for the next token in the target.
    """

    tgt_emb = self.decoder_positional_encoding(
        self.tgt_tok_emb(current_tgt_seq) * math.sqrt(self.d_model)
    )

    # Output shape: (B, current_tgt_len, D_model)
    decoder_output_all_steps = self.transformer.decoder(
        tgt=tgt_emb,
        memory=memory,
        tgt_mask=self._generate_causal_mask(current_tgt_seq.size(1)),
        tgt_key_padding_mask=None,  # No padding in tgt.
        memory_key_padding_mask=memory_key_padding_mask,
    )

    last_token_output = decoder_output_all_steps[:, -1, :]  # (B, D_model)
    return self.generator(last_token_output)  # (B, V)
