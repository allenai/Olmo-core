Datasets and Data loading
=========================

.. note::
   Most of this guide is specific to text-based data, however the :class:`~olmo_core.train.Trainer` can be
   used with other modalities as well by creating a custom data loader subclass of
   :class:`~olmo_core.data.data_loader.DataLoaderBase` (see `Using a custom data loader`_ below).

Using Olmo-core's builtin data loading
--------------------------------------

Data preparation
~~~~~~~~~~~~~~~~

Olmo-core's builtin data loading functionality requires you to pre-tokenize your data into 1D numpy arrays of token IDs. These arrays should include all special tokens already -- such as "end of sentence" (EOS) tokens -- except for padding tokens.

For example::

    import numpy as np

    documents = ["Hello, World!", "The quick brown fox jumped over the fence"]

    # Tokenize documents
    token_ids = []
    for doc in documents:
        token_ids.extend(tokenizer.encode(doc))

    # Write token IDs to disk
    data_mmap = np.memmap("data001.npy", mode="w+", dtype=np.uint32, shape=(len(token_ids),))
    data_mmap[:] = token_ids
    data_mmap.flush()

.. seealso::
    The `dolma <https://github.com/allenai/dolma>`_ project includes an optimized toolkit for pre-processing data into this format.


Train data loading
~~~~~~~~~~~~~~~~~~

Once your data is pre-processed as above there are several different strategies available for loading that data for training.
The built-in data loading strategies can be broadly categorized into two types: fixed sequence length (FSL) training and variable sequence length (VSL) training.

You can select between FSL and VSL training by choosing the appropriate data loader class to pass to your :class:`~olmo_core.train.Trainer`.
For example, the :class:`~olmo_core.data.data_loader.NumpyFSLDataLoader` or :class:`~olmo_core.data.composable.ComposableDataLoader` can be used for FSL training,
while the :class:`~olmo_core.data.data_loader.NumpyVSLDataLoader` can be used for VSL training.

The ``Numpy*DataLoader`` variants take a dataset that's a subclass of :class:`~olmo_core.data.numpy_dataset.NumpyDatasetBase`,
which handles the details of loading and sampling from your pre-tokenized numpy data files,
while the :class:`~olmo_core.data.composable.ComposableDataLoader` takes one or more :class:`~olmo_core.data.composable.InstanceSource` objects.

The rest of this section will focus on the ``Numpy*Dataset`` classes, but see the :mod:`olmo_core.data.composable` module documentation to learn
more about the composable data loading API.

Numpy fixed sequence length (FSL) datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following datasets are for fixed sequence length (FSL) training with :class:`~olmo_core.data.data_loader.NumpyFSLDataLoader`, where every training instance is exactly the same length (``sequence_length``), possibly
with document fragmentation across instances or padding within instances. They implement different strategies for how to create those training instances from your pre-tokenized numpy data files
where the sequence lengths of individual documents may vary widely.

Concatenate and chunk (:class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset` or :class:`~olmo_core.data.numpy_dataset.NumpyFSLDatasetMixture`):
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The simplest strategy is the "concatenate and chunk" approach, which means all tokenized documents are concatenated together and then chunked into training instances of the desired sequence length.
To use this method you just need to pass a :class:`~olmo_core.data.data_loader.NumpyFSLDataLoader` with a :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset` to your trainer.

While this strategy is simple and efficient, it does have a couple downsides:

1. Documents often end up fragmented across multiple training instances. That is, the beginning of a document may be in one training instance while the end of the same document is in another training instance (and thus the model will not be allowed to attend to the entire document at once).
2. Since each training instance may be composed of multiple documents, the model will be attending to tokens across more than one document simultaneously, which could potentially have adverse affects on the model (see `Zhao et al. (2024) <Zhao et al 2024_>`_ for example).

Alternatively, you can use :class:`~olmo_core.data.numpy_dataset.NumpyFSLDatasetMixture` to create a dataset that is a fine-grained mixture
of dataset sources. See :ref:`the dataset mixing guide <data_mixing>` for more details.

Concatenate and chunk + intra-document masking (:class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset` w/ ``generate_doc_lengths=True``):
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Downside #2 from above can be addressed by using `intra-document masking <intra-document masking_>`_, which you can enable
by setting ``generate_doc_lengths=True`` in your :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset` and using
a model that accepts the parameters ``doc_lens`` and ``max_doc_lens`` to its ``forward()`` method. It is expected
that if ``doc_lens`` and ``max_doc_lens`` are provided, the model will apply intra-document masking internally.

See the :class:`~olmo_core.nn.transformer.Transformer` model implementation for an example.

Document packing (:class:`~olmo_core.data.numpy_dataset.NumpyPackedFSLDataset`):
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

An alternative to the concatenate and chunk approach that also addresses the issue of document fragmentation is `document packing <https://arxiv.org/pdf/2404.10830>`_,
which uses the Optimized Best-Fit Decreasing (OBFD) bin-packing algorithm to pack documents into instances without
fragmentation (except for sequences longer than the dataset's ``sequence_length``) and with minimal padding.

Long documents can be handled by either truncating (and discarding) excess tokens or falling back to fragmenting across instances. See :class:`~olmo_core.data.types.LongDocStrategy`.

By default, OBFD is applied to each source file separately, which typically achieves very good compactness (minimal padding) if the ``.npy`` source files are
large enough (>1 Gb), and also allows for parallelization of the packing process, which can be somewhat time consuming at the start of training.
You can optionally pack instances from multiple consecutive source files together by setting the ``source_group_size`` parameter to a value greater than 1.

Document padding (:class:`~olmo_core.data.numpy_dataset.NumpyPaddedFSLDataset`):
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
This strategy creates fixed-length training instances by padding each document to the target sequence length.
Documents shorter than ``sequence_length`` are padded with padding tokens, while documents longer than
``sequence_length`` are fragmented into multiple instances.

This approach ensures that tokens from different documents never appear in the same training instance,
avoiding cross-document attention without requiring intra-document masking. However, it can be inefficient
if your documents vary widely from ``sequence_length``, as many padding tokens may be needed. In general,
using :class:`~olmo_core.data.numpy_dataset.NumpyPackedFSLDataset` is preferred over this approach.

Interleaved documents (:class:`~olmo_core.data.numpy_dataset.NumpyInterleavedFSLDataset`):
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This dataset will form instances by chunking documents and then interleaving these chunks. The purpose of this approach is to
force the model to attend to tokens that are far apart in the training instance, which may help with long-range dependencies.
Does not support intra-document masking as that would largely defeat the purpose of interleaving.

Numpy variable sequence length (VSL) training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The natural alternative to FSL is variable sequence length (VSL) training. You can use this approach by setting
a :class:`~olmo_core.data.data_loader.NumpyVSLDataLoader` as your trainer's :data:`~olmo_core.train.Trainer.data_loader`.

There is only one built-in dataset for VSL training: :class:`~olmo_core.data.numpy_dataset.NumpyVSLDataset`.
This dataset is used to inject a sequence length-based curriculum during training as introduced in
`Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
<https://arxiv.org/pdf/2405.13226>`_.

When using dataset decomposition, every training instance is a unique subset of tokens from a single document. Therefore there's no need for intra-document masking.

Using :class:`~olmo_core.data.numpy_dataset.NumpyVSLDataset` requires you set a ``min_sequence_length`` and ``max_sequence_length``
which must both be powers of 2 (e.g. 256 and 4096). Each training batch will be composed of instances of the same sequence length
such that the total number of tokens in the batch is equal to your :data:`~olmo_core.train.Trainer.global_batch_size`. Your model
must be able to handle sequences of up to ``max_sequence_length``.

You can configure a :class:`~olmo_core.data.numpy_dataset.VSLCurriculum` to control the sampling probability of different sequence lengths over the course of an epoch.

Using a custom data loader
--------------------------

Using a custom data loader with the :class:`~olmo_core.train.Trainer` just requires implementing your own :class:`~olmo_core.data.data_loader.DataLoaderBase` subclass.

In particular you need to take care when implementing the :meth:`~olmo_core.data.data_loader.DataLoaderBase._iter_batches` method to ensure each batch returned only contains the rank's local portion of the batch (with exactly :data:`~olmo_core.data.data_loader.DataLoaderBase.rank_batch_size` tokens).
You should also implement :meth:`~olmo_core.data.data_loader.DataLoaderBase.state_dict()` and :meth:`~olmo_core.data.data_loader.DataLoaderBase.load_state_dict()` such that your data loader will pick up where it left off in an epoch after :meth:`~olmo_core.data.data_loader.DataLoaderBase.load_state_dict()` is called.

For a real-world example see the source code of :class:`~olmo_core.data.data_loader.NumpyDataLoaderBase`.
But for simplicity here's a toy example that just generates random token IDs.

.. literalinclude:: ../../../src/test/data/custom_data_loader.py
   :language: py

.. _intra-document masking: https://www.semanticscholar.org/paper/c4673ed74c1f25a67c346dfddb4944e64b0d00a6
.. _Zhao et al 2024: `intra-document masking`_

.. _dataset decomposition: https://www.semanticscholar.org/paper/a14648abc56c602396634609e911f0ec071e43c1
.. _Pouransari et al 2024: `dataset decomposition`_
