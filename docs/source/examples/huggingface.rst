HuggingFace models
==================

The Olmo-core :class:`~olmo_core.train.Trainer` can be used to fine-tune language models from HuggingFace's ``transformers`` library.

One way to do this would be to manually apply a data parallel wrapper (like DDP or FSDP) to your ``AutoModelForCausalLM`` and then pass that model directly to the trainer. The downside with this approach is that you won't be able to take advantage of all of the optimizations in this library.

Instead we recommend converting your HuggingFace checkpoint into a format that can be loaded into an equivalent Olmo-core :class:`~olmo_core.nn.transformer.Transformer` model, when possible, using the functions provided by :mod:`olmo_core.nn.hf`.

Below is an example that shows how to convert an OLMo2 or Llama-3 checkpoint on HuggingFace into the right format for Olmo-core, and an example for how to convert Olmo-core checkpoints into HuggingFace formats. The mapping of OLMo Core and HF states can be configured using the constants in :mod:`olmo_core.nn.hf.convert` (see :mod:`olmo_core.nn.hf`).

.. seealso::
   See the `train a Llama model <llama.html>`_ example to learn how to use Olmo-core's training API to pretrain or fine-tune any Llama-like language model.

.. tab:: ``src/examples/huggingface/convert_checkpoint_from_hf.py``

   .. literalinclude:: ../../../src/examples/huggingface/convert_checkpoint_from_hf.py
      :language: py

.. tab:: ``src/examples/huggingface/convert_checkpoint_to_hf.py``

   .. literalinclude:: ../../../src/examples/huggingface/convert_checkpoint_to_hf.py
      :language: py

Hybrid MoE export
-----------------

The ``olmo3moe`` HF implementation supports hybrid KDA/full-attention models with optional
latent expert projections and EMO metadata, independent per-head Q/K gains, and scalable
softmax. At inference, EMO must expose all experts; restricted inference pools are rejected.
The exporter validates configuration compatibility and performs an exact tensor round trip
before writing the checkpoint. The inverse conversion uses temporary expert tensors, so allow
additional host memory when exporting large checkpoints.

The standalone HF KDA implementation requires ``use_cache=False`` and one unpadded document
per input row. Packed document boundaries and reset position IDs are rejected. Recurrent cached
generation is supplied by the separate scaling-ladders vLLM plugin. Olmo-core's scalable-softmax
attention currently rejects context parallelism, sliding-window attention, and KV-cache setup.

Tokenizer export prefers an explicit tokenizer override, then a tokenizer saved beside the
checkpoint, then the identifier in the saved training config. Its serialized backend is preserved
and checked after reload. Explicit training special-token IDs take precedence; an unspecified
BOS leaves the source tokenizer's BOS unchanged. For SFT, use the saved training tokenizer and
chat template.
