Native Generation and Chat
==========================

Olmo-core includes a native generation module for autoregressive text generation with transformer models.
This guide covers how to load a model from a checkpoint, generate text programmatically, and use the
built-in interactive chat interface.

Loading a model from a checkpoint
---------------------------------

The simplest way to get started is with :meth:`~olmo_core.generate.TransformerGenerationModule.from_checkpoint`,
which loads a transformer model and its weights from a checkpoint directory:

.. code-block:: python

    from olmo_core.generate import GenerationConfig, TransformerGenerationModule

    generation_config = GenerationConfig(
        pad_token_id=0,
        eos_token_id=1,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

    generation_module = TransformerGenerationModule.from_checkpoint(
        checkpoint_dir="path/to/checkpoint",
        generation_config=generation_config,
    )

The checkpoint must contain a ``config.json`` with the model architecture (``model`` key) and tokenizer
config (``dataset.tokenizer`` key). If your checkpoint doesn't include a ``config.json``, you can pass
the :class:`~olmo_core.nn.transformer.TransformerConfig` explicitly:

.. code-block:: python

    from olmo_core.nn.transformer import TransformerConfig

    generation_module = TransformerGenerationModule.from_checkpoint(
        checkpoint_dir="path/to/checkpoint",
        transformer_config=TransformerConfig.olmo2_7B(),
        generation_config=generation_config,
    )

You can also control the model dtype and attention backend:

.. code-block:: python

    from olmo_core.config import DType
    from olmo_core.nn.attention import AttentionBackendName

    generation_module = TransformerGenerationModule.from_checkpoint(
        checkpoint_dir="path/to/checkpoint",
        generation_config=generation_config,
        dtype=DType.bfloat16,
        attention_backend=AttentionBackendName.torch,
    )

Generating text
---------------

Use :meth:`~olmo_core.generate.TransformerGenerationModule.generate_batch` to generate token IDs from
input prompts:

.. code-block:: python

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

    # Encode a prompt
    input_ids = tokenizer.encode("The capital of France is", return_tensors="pt")

    # Generate
    generated_ids, logits, logprobs = generation_module.generate_batch(
        input_ids,
        completions_only=True,
    )

    # Decode
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(text)

``generate_batch`` returns a tuple of ``(generated_ids, logits, logprobs)``. By default ``logits``
and ``logprobs`` are ``None``; set ``return_logits=True`` or ``return_logprobs=True`` to include them.

Setting ``completions_only=True`` returns only the newly generated tokens (excluding the prompt).

Batched generation
~~~~~~~~~~~~~~~~~~

``generate_batch`` accepts batched inputs. When prompts have different lengths, use left-padding and
pass an attention mask:

.. code-block:: python

    prompts = ["Hello, world!", "The quick brown fox"]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )

    generated_ids, _, _ = generation_module.generate_batch(
        encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        completions_only=True,
    )

    for i, ids in enumerate(generated_ids):
        print(f"Prompt: {prompts[i]}")
        print(f"Output: {tokenizer.decode(ids, skip_special_tokens=True)}")

Generation configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~olmo_core.generate.GenerationConfig` controls how tokens are selected. Key parameters:

- ``max_new_tokens`` / ``max_length`` -- limits the number of generated tokens.
- ``do_sample`` -- set to ``False`` for greedy (deterministic) decoding.
- ``temperature`` -- higher values produce more random outputs; ``0.0`` is equivalent to greedy.
- ``top_k`` -- restrict sampling to the top-k highest-probability tokens (``-1`` disables).
- ``top_p`` -- nucleus sampling; only consider tokens whose cumulative probability exceeds this threshold.
- ``use_cache`` -- enable KV-cache for faster autoregressive decoding (enabled by default).
- ``stop_token_ids`` -- additional token IDs (beyond EOS) that stop generation.

You can override any generation parameter per call:

.. code-block:: python

    # Greedy decoding for this call only
    generated_ids, _, _ = generation_module.generate_batch(
        input_ids,
        do_sample=False,
    )

Using the config-based API
~~~~~~~~~~~~~~~~~~~~~~~~~~

For more structured setups, use :class:`~olmo_core.generate.TransformerGenerationModuleConfig`:

.. code-block:: python

    from olmo_core.generate import GenerationConfig, TransformerGenerationModuleConfig

    config = TransformerGenerationModuleConfig(
        generation_config=GenerationConfig(
            pad_token_id=0,
            eos_token_id=1,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.95,
        ),
        compile_model=True,
    )

    generation_module = config.build(
        checkpoint_dir="path/to/checkpoint",
    )

Merging multiple checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~olmo_core.generate.TransformerGenerationModule.from_checkpoints` averages the weights from
multiple checkpoints before creating the generation module:

.. code-block:: python

    generation_module = TransformerGenerationModule.from_checkpoints(
        checkpoint_dirs=[
            "path/to/checkpoint1",
            "path/to/checkpoint2",
            "path/to/checkpoint3",
        ],
        generation_config=generation_config,
    )

Interactive chat interface
--------------------------

Olmo-core ships with a CLI chatbot that wraps the generation module in an interactive loop
with conversation history and chat template support.

Basic usage
~~~~~~~~~~~

.. code-block:: bash

    python -m olmo_core.generate.chat path/to/checkpoint

This loads the model, auto-detects the tokenizer from the checkpoint's ``config.json``, and starts
an interactive prompt.

Running on Mac (no Flash Attention)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're running on a Mac (e.g. Apple Silicon) without Flash Attention, use the ``torch`` attention
backend and disable the KV cache:

.. code-block:: bash

    python -m olmo_core.generate.chat path/to/checkpoint \
        --attention-backend torch --no-use-cache

You can also use a public checkpoint URL directly:

.. code-block:: bash

    python -m olmo_core.generate.chat \
        https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage2/step47684/ \
        --attention-backend torch --no-use-cache

Customizing generation parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m olmo_core.generate.chat path/to/checkpoint \
        --max-new-tokens 512 \
        --temperature 0.7 \
        --top-p 0.9

    # Greedy decoding
    python -m olmo_core.generate.chat path/to/checkpoint \
        --no-do-sample

Chat templates
~~~~~~~~~~~~~~

By default the chat interface concatenates messages without any special formatting. For models
trained with a chat template (e.g. instruction-tuned models), pass a Jinja2 template string
via ``--chat-template``:

.. code-block:: bash

    python -m olmo_core.generate.chat path/to/checkpoint \
        --chat-template "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}{% endfor %}<|assistant|>"

System prompts
~~~~~~~~~~~~~~

Provide a system prompt that is prepended to every conversation:

.. code-block:: bash

    python -m olmo_core.generate.chat path/to/checkpoint \
        --system-prompt "You are a helpful assistant."

In-chat commands
~~~~~~~~~~~~~~~~

While in the chat session:

- ``/quit`` or ``/exit`` -- exit the chatbot
- ``/clear`` -- clear conversation history
- ``/help`` -- show help

All CLI options
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``checkpoint_dir``
     - (required)
     - Path or URL to model checkpoint
   * - ``--max-new-tokens``
     - 1024
     - Maximum tokens to generate per turn
   * - ``--max-length``
     - None
     - Maximum total length (prompt + generation); overrides ``--max-new-tokens``
   * - ``--temperature``
     - 1.0
     - Sampling temperature
   * - ``--top-k``
     - -1
     - Top-k filtering (-1 = disabled)
   * - ``--top-p``
     - 0.7
     - Nucleus sampling threshold
   * - ``--do-sample / --no-do-sample``
     - True
     - Enable/disable sampling
   * - ``--use-cache / --no-use-cache``
     - True
     - Enable/disable KV cache
   * - ``--attention-backend``
     - auto
     - Attention backend (e.g. ``torch``, ``flash``)
   * - ``--dtype``
     - bfloat16
     - Model parameter dtype
   * - ``--system-prompt``
     - None
     - System prompt for the conversation
   * - ``--show-special-tokens``
     - False
     - Show special tokens in output
   * - ``--chat-template``
     - concatenate
     - Jinja2 chat template string
   * - ``--verbosity``
     - WARNING
     - Logging level
