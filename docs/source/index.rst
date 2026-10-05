.. olmo-core documentation master file, created by
   sphinx-quickstart on Tue Sep 21 08:07:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**Olmo-core**
===============

**Olmo-core** is a Python library that provides building blocks for large-scale distributed training with PyTorch.

To get started first install `PyTorch <https://pytorch.org>`_ according to the official instructions
specific to your environment. Then you can install Olmo-core from PyPI with:

.. code-block:: bash

    pip install ai2-olmo-core

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Overview

   overview/introduction.rst
   overview/installation.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Guides

   guides/all_in_one_for_researchers.md
   guides/data_loading.rst
   guides/data_mixing.rst
   guides/generation.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/huggingface.rst
   examples/llm.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   config
   data/index
   distributed/index
   eval/index
   exceptions
   float8
   fs_cache
   generate/index
   io
   launch
   nn/index
   optim
   testing
   train/index
   utils

.. toctree::
   :hidden:
   :caption: Development

   License <https://raw.githubusercontent.com/allenai/Olmo-core/main/LICENSE>
   CHANGELOG <https://github.com/allenai/Olmo-core/blob/main/CHANGELOG.md>
   GitHub Repository <https://github.com/allenai/Olmo-core>

Team
----

**Olmo-core** is developed and maintained at
`the Allen Institute for Artificial Intelligence (AI2) <https://allenai.org/>`_.
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.

To learn more about who specifically contributed to this codebase, see
`our contributors <https://github.com/allenai/Olmo-core/graphs/contributors>`_ page.

License
-------

**Olmo-core** is licensed under `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
A full copy of the license can be found `on GitHub <https://github.com/allenai/Olmo-core/blob/main/LICENSE>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
