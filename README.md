# RegressLM: Easy Text-to-Text Regression
[![Continuous Integration](https://github.com/google-deepmind/regress_lm/actions/workflows/core_test.yml/badge.svg)](https://github.com/google-deepmind/regress_lm/actions?query=branch%3Amain)

## Overview
RegressLM is a library for text-to-text regression, applicable to any input
string representation and allows pretraining and fine-tuning over multiple
regression tasks.

## General Usage
See tests for specific API usage. The overall use-case is to import a RegressLM
model, which can decode floating-point predictions from a given input, and also
gradient update against new data.

```python
from regress_lm import core
from regress_lm import models
from regress_lm import tokenizers
from regress_lm import vocabs

# Define input string tokenization
encoder_vocab = vocabs.SentencePieceVocab.from_t5()

# Define float tokenization
decoder_tokenizer = tokenizers.P10Tokenizer()
decoder_vocab = vocabs.DecoderVocab(decoder_tokenizer)

# Create RegressLM model with max input token lengths.
model = models.PyTorchModel(encoder_vocab, decoder_vocab, max_input_len=2048)

# Example (x,y) pairs, which can be fine-tuned against.
examples = [core.Example(x='hello', y=0.3), core.Example(x='world', y=-0.3)]
examples_t = model.convert_examples(examples)
model.update(examples_t)

# Query inputs.
inputs = [core.ExampleInput(x='hi')]
inputs_t = model.convert_inputs(inputs)
tokens, floats = model.decode(inputs_t, num_samples=128)
```

**Disclaimer:** This is not an officially supported Google product.