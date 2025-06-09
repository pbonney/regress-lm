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

"""PyTorch implementation of a RegressLM."""

from typing import Iterable
import numpy as np
from regress_lm import core
from regress_lm import vocabs
from regress_lm.models import base as model_base
from regress_lm.models.pytorch import architecture
import torch
from torch import optim
import torch.nn.functional as F

NEG_INF = -1.0e7

# Dict Keys: "encoder_input", "decoder_input", "decoder_target"
Tensor = torch.Tensor


class PyTorchModel(model_base.Model[Tensor]):
  """PyTorch implementation of a RegressLM."""

  def __init__(
      self,
      encoder_vocab: vocabs.EncoderVocab[str],
      decoder_vocab: vocabs.DecoderVocab[float],
      max_input_len: int = 2048,
      learning_rate: float = 1e-4,
      **architecture_kwargs,
  ):
    self.max_input_len = max_input_len
    self.learning_rate = learning_rate

    self.encoder_vocab = encoder_vocab
    self.decoder_vocab = decoder_vocab

    self.encoder_decoder = architecture.EncoderDecoder(
        encoder_vocab=self.encoder_vocab,
        decoder_vocab=self.decoder_vocab,
        max_encoder_len=self.max_input_len,
        **architecture_kwargs,
    )

    self.optimizer = optim.Adafactor(
        filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()),
        lr=self.learning_rate,
    )

    # Pre-compute the constraint masks for the decoder.
    self.decoder_constraint_masks = self._create_decoder_constraint_masks()

  def update(self, xy: dict[str, Tensor]) -> dict[str, Tensor]:
    self.encoder_decoder.train()
    self.optimizer.zero_grad()

    logits = self.encoder_decoder.forward(
        xy['encoder_input'], xy['decoder_input']
    )
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),  # (B * L_decode, V)
        xy['decoder_target'].reshape(-1),  # Reshape to (B * L_decode)
        ignore_index=self.decoder_vocab.bos_pad_id,
    )

    loss.backward()
    self.optimizer.step()
    return {'loss': loss.mean()}

  def pretrain(self, ds: Iterable[dict[str, Tensor]]) -> dict[str, Tensor]:
    raise NotImplementedError()

  @torch.no_grad()
  def decode(
      self, inputs: dict[str, Tensor], num_samples: int  # (B, L_src)
  ) -> tuple[Tensor, np.ndarray]:
    self.encoder_decoder.eval()
    encoder_input = inputs['encoder_input']
    batch_size = encoder_input.shape[0]
    bos_pad_id = self.decoder_vocab.bos_pad_id
    # memory: (B, L_src, D_model), memory_key_padding_mask: (B, L_src)
    memory, memory_key_padding_mask = self.encoder_decoder.encode(encoder_input)

    # Expand/Repeat encoder outputs and masks for num_samples
    # Effectively, new batch_size = B * num_samples
    # memory: (B, L_src, D) -> (B, 1, L_src, D) -> (B, S, L_src, D)
    # -> (B*S, L_src, D)
    expanded_memory = (
        memory.unsqueeze(1)
        .repeat(1, num_samples, 1, 1)
        .view(batch_size * num_samples, memory.size(1), memory.size(2))
    )
    expanded_memory_key_padding_mask = (
        memory_key_padding_mask.unsqueeze(1)
        .repeat(1, num_samples, 1)
        .view(batch_size * num_samples, memory_key_padding_mask.size(1))
    )

    # Initialize decoder input for the expanded batch, start with <pad>.
    current_tgt_ids = torch.full(
        (batch_size * num_samples, 1), bos_pad_id, dtype=torch.long
    )

    # Store all generated token IDs for all sequences in the expanded batch
    decode_len = self.decoder_vocab.decode_len
    generated_sequences_ids = torch.zeros(
        (batch_size * num_samples, decode_len), dtype=torch.long
    )
    # Store log probabilities for each generated sequence in the expanded batch
    sequences_log_probs = torch.zeros(
        batch_size * num_samples, dtype=torch.float32
    )

    # Batched autoregressive decoding loop
    for step_idx in range(decode_len):
      # Get logits for the next token for all (B * num_samples) sequences
      # Shape: (B*S, V)
      logits = self.encoder_decoder.next_token_logits(
          current_tgt_ids, expanded_memory, expanded_memory_key_padding_mask
      )

      # Apply constraints using the pre-computed mask
      curr_mask = self.decoder_constraint_masks[step_idx, :]  # (V,)
      curr_mask = curr_mask.unsqueeze(0)  # (1, V)
      masked_logits = (1.0 - curr_mask) * NEG_INF + curr_mask * logits

      # Apply temperature sampling
      temperature = 1.0
      # Sample 1 token for each of the B*S sequences
      probs = F.softmax(masked_logits / temperature, dim=-1)
      # (B*S,)
      token_ids_expanded_batch = torch.multinomial(probs, num_samples=1)
      token_ids_expanded_batch = token_ids_expanded_batch.squeeze(-1)

      # Get log probabilities of the chosen tokens
      # (B*S, V)
      current_step_log_probs_dist = F.log_softmax(masked_logits, dim=-1)
      # (B*S,)
      chosen_token_log_probs = torch.gather(
          current_step_log_probs_dist,
          1,
          token_ids_expanded_batch.unsqueeze(-1),
      )
      chosen_token_log_probs = chosen_token_log_probs.squeeze(-1)
      sequences_log_probs += chosen_token_log_probs  # Accumulate

      # Store the predicted token IDs
      generated_sequences_ids[:, step_idx] = token_ids_expanded_batch

      # Prepare input for the next step, but only if not the last float token
      if step_idx < decode_len - 1:
        current_tgt_ids = torch.cat(
            [current_tgt_ids, token_ids_expanded_batch.unsqueeze(-1)],
            dim=1,
        )

    # Reshape outputs back to (B, num_samples, L_decode)
    final_decoded_ids = generated_sequences_ids.view(
        batch_size, num_samples, decode_len
    )

    # Compute equivalent floats.
    output_floats = np.zeros((batch_size, num_samples), dtype=float)
    for b in range(batch_size):
      for s_idx in range(num_samples):
        output_floats[b, s_idx] = self.decoder_vocab.from_token_ids(
            final_decoded_ids[b, s_idx, :].tolist()
        )

    return final_decoded_ids, output_floats

  def log_prob(self, example: dict[str, Tensor]) -> Tensor:
    self.encoder_decoder.eval()
    enc_input = example['encoder_input']
    dec_input = example['decoder_input']
    dec_target = example['decoder_target']

    logits = self.encoder_decoder.forward(enc_input, dec_input)
    log_probs = F.log_softmax(logits, dim=-1)

    true_log_probs = torch.gather(
        log_probs, dim=2, index=dec_target.unsqueeze(-1)
    )

    pad_mask = dec_target != self.decoder_vocab.bos_pad_id
    true_log_probs_masked = true_log_probs.squeeze(-1) * pad_mask
    sequence_sum_log_probs = true_log_probs_masked.sum(dim=1)
    return sequence_sum_log_probs.detach()

  def convert_inputs(
      self, inputs: list[core.ExampleInput]
  ) -> dict[str, Tensor]:
    strings = [example.x for example in inputs]
    encoder_token_ids = [self.encoder_vocab.to_token_ids(s) for s in strings]
    encoder_input = [self._pad_or_truncate(t) for t in encoder_token_ids]
    return {'encoder_input': torch.tensor(encoder_input)}

  def convert_examples(self, examples: list[core.Example]) -> dict[str, Tensor]:
    y_values = [example.y for example in examples]
    decoder_token_ids = [self.decoder_vocab.to_token_ids(y) for y in y_values]
    decoder_input = [self._pad_front(t) for t in decoder_token_ids]
    decoder_target = [self._pad_last(t) for t in decoder_token_ids]

    out = self.convert_inputs(examples)
    out.update({
        'decoder_input': torch.tensor(decoder_input),
        'decoder_target': torch.tensor(decoder_target),
    })
    return out

  def _pad_or_truncate(self, token_ids: list[int]) -> list[int]:
    encoder_pad_idx = self.encoder_vocab.pad_id
    if len(token_ids) > self.max_input_len:
      return token_ids[: self.max_input_len]
    return token_ids + [encoder_pad_idx] * (self.max_input_len - len(token_ids))

  def _pad_last(self, token_ids: list[int]) -> list[int]:
    return token_ids + [self.decoder_vocab.bos_pad_id]

  def _pad_front(self, token_ids: list[int]) -> list[int]:
    return [self.decoder_vocab.bos_pad_id] + token_ids

  def _create_decoder_constraint_masks(self) -> torch.Tensor:
    num_float_tokens = self.decoder_vocab.decode_len
    vocab_size = len(self.decoder_vocab)

    logits_masks_np = np.zeros((num_float_tokens, vocab_size), dtype=np.float32)
    for step_idx in range(num_float_tokens):
      for allowed_token_id in self.decoder_vocab.token_ids_at_index(step_idx):
        logits_masks_np[step_idx, allowed_token_id] = 1.0

    return torch.from_numpy(logits_masks_np)
