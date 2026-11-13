Train an LLM
============

The following snippets can be found in `src/examples/llm/ <https://github.com/allenai/Olmo-core/tree/main/src/examples/llm>`_.
The ``train.py`` script is meant to be launched via ``torchrun``.
You can also use the ``python -m olmo_core.launch.beaker`` CLI to quickly launch this script on Beaker.

.. tab:: ``train.py``

   .. literalinclude:: ../../../src/examples/llm/train.py
      :language: py
