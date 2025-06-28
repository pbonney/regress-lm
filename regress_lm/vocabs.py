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

"""Custom vocab classes for RegressLM."""

import abc
import collections
from typing import Generic, TypeVar
from regress_lm import tokenizers
import torchtext as tt
import sentencepiece as sp

ObjectT = TypeVar('ObjectT')


class BaseVocab(abc.ABC, Generic[ObjectT]):
  """Base class for vocabularies."""

  @abc.abstractmethod
  def to_token_ids(self, obj: ObjectT, /) -> list[int]:
    """Converts object (e.g. text) to token ids."""

  @abc.abstractmethod
  def __len__(self) -> int:
    """Returns the vocab size."""


class EncoderVocab(BaseVocab[ObjectT]):
  """Vocabulary class for encoders.

  Note we don't ever need to convert back to text.
  """

  @property
  @abc.abstractmethod
  def pad_id(self) -> int:
    """Returns the pad id."""


class DecoderVocab(BaseVocab[ObjectT]):
  """Vocabulary class for decoders."""

  def __init__(self, tokenizer: tokenizers.DecoderTokenizer[ObjectT]):
    self.tokenizer = tokenizer
    self.tt_vocab = tt.vocab.Vocab(
        counter=collections.Counter(
            {t: 1 for t in self.tokenizer.all_tokens()}
        ),
        specials=['<pad>'],
    )

  def to_token_ids(self, obj: ObjectT, /) -> list[int]:
    """Converts object to token ids."""
    return [self.tt_vocab[t] for t in self.tokenizer.to_tokens(obj)]

  def from_token_ids(self, token_ids: list[int], /) -> ObjectT:
    """Converts token ids to object."""
    token_strs = [self.tt_vocab.itos[id] for id in token_ids]
    return self.tokenizer.from_tokens(token_strs)

  def token_ids_at_index(self, index: int) -> list[int]:
    """Returns the token ids for the given index."""
    return [self.tt_vocab[t] for t in self.tokenizer.tokens_at_index(index)]

  @property
  def bos_pad_id(self) -> int:
    """Returns the BOS / PAD id for the decoder."""
    return self.tt_vocab['<pad>']

  @property
  def decode_len(self) -> int:
    """Returns the number of tokens used to represent each float."""
    return self.tokenizer.num_tokens_per_obj

  def __len__(self) -> int:
    """Returns the vocab size."""
    return len(self.tt_vocab)


class BasicEnglishVocab(EncoderVocab[str]):
  """Basic English vocab for testing."""

  def __init__(self, words: list[str]):
    self.tokenizer = tt.data.utils.get_tokenizer('basic_english')
    self.vocab = tt.vocab.Vocab(
        collections.Counter({w: 1 for w in words}), specials=['<pad>', '<unk>']
    )

  def to_token_ids(self, obj: str) -> list[int]:
    tokenized = self.tokenizer(obj)
    return [self.vocab[t] for t in tokenized]

  @property
  def pad_id(self) -> int:
    return self.vocab['<pad>']

  def __len__(self) -> int:
    return len(self.vocab)


class SentencePieceVocab(EncoderVocab[str]):
  """SentencePiece vocab."""

  T5_FILE = 'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model'

  def __init__(self, file_path: str):
    """Initializes SentencePieceVocab by loading a pre-trained .model file."""
    self.sp_processor = sp.SentencePieceProcessor()
    self.sp_processor.Load(file_path)

    if self.sp_processor.pad_id() == -1:
      raise ValueError(
          f"SentencePiece model '{file_path}' does not have a PAD token"
          ' explicitly defined.'
      )

  def to_token_ids(self, obj: str, /) -> list[int]:
    """Converts text to a list of token ids using the SentencePiece model."""
    return self.sp_processor.EncodeAsIds(obj)

  @property
  def pad_id(self) -> int:
    """Returns the pad id defined in the SentencePiece model."""
    return self.sp_processor.pad_id()

  def __len__(self) -> int:
    """Returns the total vocabulary size."""
    return self.sp_processor.GetPieceSize()

  @classmethod
  def from_t5(cls) -> 'SentencePieceVocab':
    return cls(cls.T5_FILE)
