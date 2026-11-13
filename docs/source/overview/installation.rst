Installation
============

Prior to installing Olmo-core you should install `PyTorch <https://pytorch.org>`_ according to the official instructions
specific to your operating system and hardware.

Then you can install Olmo-core from `PyPI <https://pypi.org/project/ai2-olmo-core/>`_ with::

    pip install ai2-olmo-core

There are a number of optional dependencies that must be installed to use certain functionality as well, including:

- `flash-attn <https://github.com/Dao-AILab/flash-attention>`_, `ring-flash-attn <https://github.com/zhuzilin/ring-flash-attention>`_, and `TransformerEngine <https://github.com/NVIDIA/TransformerEngine>`_ for the corresponding attention backends.
- `Liger-Kernel <https://github.com/linkedin/Liger-Kernel>`_ for a low-memory "fused-linear" loss implementation.
- `torchao <https://github.com/pytorch/ao>`_ for float8 training (see :mod:`olmo_core.float8`).
- `grouped_gemm <https://github.com/tgale96/grouped_gemm>`_ for dropless mixture-of-experts (MoE) models (see :mod:`olmo_core.nn.moe`).
