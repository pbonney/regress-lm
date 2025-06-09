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

"""Base class for a RegressLM model."""

import abc
from typing import Generic, Iterable, TypeVar
import numpy as np
from regress_lm import core

# Low-level tensor type.
TensorT = TypeVar('TensorT')


class Model(Generic[TensorT], abc.ABC):
  """Abstract class for a Model.

  Uses generic types to allow different low-level tensor packages (Jax, PyTorch,
  etc.). ExampleT can be jax.Array, torch.Tensor, etc.

  Conversion between high-level and low-level implementations can be done via
  wrappers.
  """

  @abc.abstractmethod
  def update(self, xy: dict[str, TensorT]) -> dict[str, TensorT]:
    """Updates the regressor with low-level gradient logic on few examples.

    Args:
      xy: Dict of tensors to be used for training.

    Returns:
      Dict of tensors for metrics and auxiliary outputs.
    """

  @abc.abstractmethod
  def pretrain(self, ds: Iterable[dict[str, TensorT]]) -> dict[str, TensorT]:
    """Pretrains the model with a dataset.

    Similar in spirit to calling update() for multiple iterations, but may be
    implemented differently to absorb lots of data.

    Args:
      ds: Iterable of examples to pretrain on.

    Returns:
      Dict of tensors for metrics and auxiliary outputs.
    """

  @abc.abstractmethod
  def decode(
      self, inputs: dict[str, TensorT], num_samples: int
  ) -> tuple[TensorT, np.ndarray]:
    """Decodes tokens and returns them and corresponding floats."""

  @abc.abstractmethod
  def log_prob(self, example: dict[str, TensorT]) -> TensorT:
    """Returns log probability of y given x."""

  @abc.abstractmethod
  def convert_inputs(
      self, inputs: list[core.ExampleInput]
  ) -> dict[str, TensorT]:
    """Converts high-level inputs to batched low-level inputs."""

  @abc.abstractmethod
  def convert_examples(
      self, examples: list[core.Example]
  ) -> dict[str, TensorT]:
    """Converts high-level examples to batched low-level examples."""
