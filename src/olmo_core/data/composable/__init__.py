"""
Overview
--------

A composable data loading API for fixed sequence length text data.

::

    ┌─────────────┐       ┌────────────────┐       ┌──────────────────────┐
    │ TokenSource │ ⇢ ⋯ ⇢ │ InstanceSource │ ⇢ ⋯ ⇢ │ ComposableDataLoader │
    └─────────────┘       └────────────────┘       └──────────────────────┘

This API consists of a series of simple, composable, elements, including:

1. :class:`TokenSource` / :class:`DocumentSource`: Token sources provide access to tokenized text data, while
   document sources are special token sources that also provide information on where the document boundaries are.
   Examples include:

   * :class:`InMemoryTokenSource` / :class:`InMemoryDocumentSource`: A simple token/document source that holds all tokens in memory.
   * :class:`ConcatenatedTokenSource` / :class:`ConcatenatedDocumentSource`: A token/document source that combines multiple sources into one.
   * :class:`SlicedTokenSource`: A token source that provides a slice into another token source.
   * :class:`NumpyDocumentSource`: A document that reads tokens from one or more numpy source files, like those created
     from the dolma toolkit.
   * :class:`SamplingTokenSource` / :class:`SamplingDocumentSource`: A token/document source that samples tokens/documents
     from one or more other token/document sources.
   * :class:`MixingTokenSource` / :class:`MixingDocumentSource`: A token/document source that mixes other token/document sources together.

2. :class:`InstanceSource`: Instance sources convert token sources (or in some case other instance sources)
   into fixed-length instances.
   Examples include:

   * :class:`ConcatAndChunkInstanceSource`: The simplest instance source that chunks up token sources
     without regard for document boundaries, just like the :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset`.
   * :class:`PackingInstanceSource`: An instance source that packs documents from one or more document
     sources into instances using an optimized packing algorithm, just like the
     :class:`~olmo_core.data.numpy_dataset.NumpyPackedFSLDataset`.
   * :class:`ConcatenatedInstanceSource`: An instance source combines instances from other instance sources.
   * :class:`SlicedInstanceSource`: An instance source that provides a slice into another instance source.
   * :class:`SamplingInstanceSource`: An instance source that samples instances from other instance sources.
   * :class:`MixingInstanceSource`: An instance source that mixes other instance sources together.
   * :class:`RandomInstanceSource`: An instance source for generating random instances.

3. :class:`ComposableDataLoader`: A data loader for Olmo-core's :class:`~olmo_core.train.Trainer` that takes
   one or more instance sources.

.. tip::
    Use :meth:`InstanceSource.visualize()` to print out a recursive visualization of an instance
    source and all its sub-sources.

Basic Examples
--------------

Create a simple instance source that chunks up in-memory token sources::

   from olmo_core.data.composable import *
   
   work_dir = "/tmp/dataset-common"
   source = ConcatAndChunkInstanceSource(
       InMemoryTokenSource(list(range(100)), work_dir=work_dir),
       sequence_length=10,
       work_dir=work_dir,
   )
   source.visualize()

::

   ConcatAndChunkInstanceSource(ee7a76d): 100 tokens
   └─ InMemoryTokenSource(73b91ee): 100 tokens

Split the source into train and test sets::

   train_source, test_source = source.split(0.8)
   train_source.visualize()
   test_source.visualize()

::

   SlicedInstanceSource(d01d0e2): 80 tokens
   └─ ConcatAndChunkInstanceSource(ee7a76d): 100 tokens
      └─ InMemoryTokenSource(73b91ee): 100 tokens
 
   SlicedInstanceSource(a5a511f): 20 tokens
   └─ ConcatAndChunkInstanceSource(ee7a76d): 100 tokens
      └─ InMemoryTokenSource(73b91ee): 100 tokens

Sample a subset of a source::

   train_source = train_source.sample(max_tokens=50)
   train_source.visualize()

::

   SamplingInstanceSource(77d8031): 50 tokens
   └─ SlicedInstanceSource(d01d0e2): 80 tokens
      └─ ConcatAndChunkInstanceSource(ee7a76d): 100 tokens
         └─ InMemoryTokenSource(73b91ee): 100 tokens

Create a mix of token sources::

   tokens1 = InMemoryTokenSource(list(range(100)), work_dir=work_dir, label="source1")
   tokens2 = InMemoryTokenSource(list(range(100, 200)), work_dir=work_dir, label="source2")
   tokens_mix = MixingTokenSource(
       MixingTokenSource.Spec(source=tokens1, ratio=0.5),
       MixingTokenSource.Spec(source=tokens2, ratio=0.5),
       work_dir=work_dir,
   )
   source = ConcatAndChunkInstanceSource(tokens_mix, sequence_length=10, work_dir=work_dir)
   source.visualize()

::

   ConcatAndChunkInstanceSource(4820826): 200 tokens
   └─ MixingTokenSource(5fc211a): 200 tokens
      ├─ SamplingTokenSource(7adca21): 100 tokens [source1]
      │  └─ InMemoryTokenSource(73b91ee): 100 tokens [source1]
      └─ SamplingTokenSource(baf2e4f): 100 tokens [source2]
         └─ InMemoryTokenSource(a9e49e1): 100 tokens [source2]

Working with numpy source files
-------------------------------

You can use the same numpy tokenized source files that the dataset classes in :mod:`olmo_core.data.numpy_dataset`
consume by starting with the :class:`NumpyDocumentSource` or its corresponding config class, :class:`NumpyDocumentSourceConfig`.

For example::

   source_config = NumpyDocumentSource.Config(
       source_paths=[
           "gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/*.npy"
       ],
       tokenizer=tokenizer,
   )
   sources = source_config.build(work_dir)

::

   [
       NumpyDocumentSource('gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/part-00-00000.npy',),
       NumpyDocumentSource('gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/part-01-00000.npy',),
       NumpyDocumentSource('gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/part-02-00000.npy',),
       NumpyDocumentSource('gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/part-03-00000.npy',),
       NumpyDocumentSource('gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/part-04-00000.npy',),
       NumpyDocumentSource('gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/part-05-00000.npy',),
       NumpyDocumentSource('gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/part-06-00000.npy',),
       NumpyDocumentSource('gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/part-07-00000.npy',),
       NumpyDocumentSource('gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/part-08-00000.npy',),
       NumpyDocumentSource('gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/part-09-00000.npy',)
   ]

Ratio-based mixing
------------------

Here's a more useful example where we create several groups of numpy sources, two code and two math,
using :meth:`NumpyDocumentSourceConfig.from_source_groups`::

   sequence_length = 8192
   token_sources = NumpyDocumentSource.Config.from_source_groups(
       {
           "code_fim": [
               "gs://ai2-llm/preprocessed/stack-edu/sample-fim-weighted-pl-edu-score-decon/**/**/*.npy",
           ],
           "swallowcode": [
               "gs://ai2-llm/preprocessed/tokyotech-llm/swallowcode/scor_final_data-decon-sparkle-motion/allenai/dolma2-tokenizer/*.npy"
           ],
           "megamath": [
               "gs://ai2-llm/preprocessed/megamath_web_pro_max/beaker_rewrites-decon-sparkle-motion/**/allenai/dolma2-tokenizer/*.npy"
           ],
           "dolminos2math": [
               "gs://ai2-llm/preprocessed/tokyotech-llm/swallowmath/beaker_outputs-decon-sparkle-motion-withids/allenai/dolma2-tokenizer/*.npy",
               "gs://ai2-llm/preprocessed/midtraining-reasoning/flat_dolmino_math-decon-sparkle-motion/allenai/dolma2-tokenizer/*.npy",
               "gs://ai2-llm/preprocessed/midtraining-reasoning/OpenMathReasoning/OpenMathReasoning-rewrite-full-thoughts/jsonls-decon-sparkle-motion/allenai/dolma2-tokenizer/*.npy",
               "gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/MIND/data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/*.npy",
               "gs://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-sparkle-motion/allenai/dolma2-tokenizer/*.npy",
           ],
       },
       tokenizer=tokenizer,
   )

And then mix them together at the instance level in a hierarchical fashion with a :class:`MixingInstanceSource`
to get a source with 30B tokens::

   def make_instance_source(label: str) -> InstanceSourceConfig:
       return ConcatAndChunkInstanceSource.Config(
           sources=[token_sources[label]], label=label, sequence_length=sequence_length
       )
   
   
   mix_config = MixingInstanceSource.Config(
       source_specs=[
           ################
           # code sources #
           ################
           MixingInstanceSource.Spec.Config(
               source=MixingInstanceSource.Config(
                   source_specs=[
                       MixingInstanceSource.Spec.Config(
                           source=make_instance_source("code_fim"),
                           ratio=0.5,
                           label="code_fim",
                       ),
                       MixingInstanceSource.Spec.Config(
                           source=make_instance_source("swallowcode"),
                           ratio=0.5,
                           label="swallowcode",
                       ),
                   ]
               ),
               ratio=0.5,
               label="code",
           ),
           ################
           # math sources #
           ################
           MixingInstanceSource.Spec.Config(
               source=MixingInstanceSource.Config(
                   source_specs=[
                       MixingInstanceSource.Spec.Config(
                           source=make_instance_source("megamath"),
                           ratio=0.1,
                           label="megamath",
                       ),
                       MixingInstanceSource.Spec.Config(
                           source=make_instance_source("dolminos2math"),
                           ratio=0.9,
                           label="dolminos2math",
                       ),
                   ]
               ),
               ratio=0.5,
               label="math",
           ),
       ],
       num_tokens=30_000_000_000,
   )

   mix = mix_config.build("/tmp/dataset-common")
   mix.visualize()

::

   MixingInstanceSource(e421147): 30.0B tokens
   ├─ SamplingInstanceSource(c65cde2): 15.0B tokens [code]
   │  └─ MixingInstanceSource(73dfe43): 37.7B tokens
   │     ├─ SamplingInstanceSource(85521bf): 18.8B tokens [code_fim]
   │     │  └─ ConcatAndChunkInstanceSource(adb4562): 21.4B tokens [code_fim]
   │     │     └─ NumpyDocumentSource x 474: 21.4B tokens [code_fim]
   │     └─ SamplingInstanceSource(8d7c840): 18.8B tokens [swallowcode]
   │        └─ ConcatAndChunkInstanceSource(b2f2ef4): 18.8B tokens [swallowcode]
   │           └─ NumpyDocumentSource x 128: 18.8B tokens [swallowcode]
   └─ SamplingInstanceSource(03941ca): 15.0B tokens [math]
      └─ MixingInstanceSource(39aa7de): 20.3B tokens
         ├─ SamplingInstanceSource(cbc20a2): 2.0B tokens [megamath]
         │  └─ ConcatAndChunkInstanceSource(2b6a324): 3.9B tokens [megamath]
         │     └─ NumpyDocumentSource x 264: 3.9B tokens [megamath]
         └─ SamplingInstanceSource(857de5e): 18.3B tokens [dolminos2math]
            └─ ConcatAndChunkInstanceSource(b768b9a): 18.3B tokens [dolminos2math]
               └─ NumpyDocumentSource x 415: 18.3B tokens [dolminos2math]

.. tip::
    The ratios (e.g. :class:`MixingInstanceSourceSpec.ratio`) for each source within a mix don't necessary need to sum to 1.0, but you'll see
    a warning if they don't and they'll be normalized before being applied::

        UserWarning: Target mixing ratios don't sum to 1. They will be normalized as follows:
         ❯ Source 'math': target ratio adjusted from 0.7 to 0.7368421052631579
         ❯ Source 'code': target ratio adjusted from 0.25 to 0.2631578947368421

Up-sampling or targeted repetition
----------------------------------

Suppose we wanted to simulate training 3 epochs on the mixture above, i.e. training on 3 repetitions
of the data.
In general you can do exact up-sampling by wrapping a source in
a :class:`SamplingInstanceSource` (or :class:`SamplingTokenSource`, :class:`SamplingDocumentSource`),
or by calling the ``.sample()`` / ``.resize()`` methods::

   upsampled_mix = mix.resize(3.0)
   upsampled_mix.visualize()

::

   SamplingInstanceSource(de59d5e): 90.0B tokens
   └─ MixingInstanceSource(e421147): 30.0B tokens
      ├─ SamplingInstanceSource(c65cde2): 15.0B tokens [code]
      │  └─ MixingInstanceSource(73dfe43): 37.7B tokens
      │     ├─ SamplingInstanceSource(85521bf): 18.8B tokens [code_fim]
      │     │  └─ ConcatAndChunkInstanceSource(adb4562): 21.4B tokens [code_fim]
      │     │     └─ NumpyDocumentSource x 474: 21.4B tokens [code_fim]
      │     └─ SamplingInstanceSource(8d7c840): 18.8B tokens [swallowcode]
      │        └─ ConcatAndChunkInstanceSource(b2f2ef4): 18.8B tokens [swallowcode]
      │           └─ NumpyDocumentSource x 128: 18.8B tokens [swallowcode]
      └─ SamplingInstanceSource(03941ca): 15.0B tokens [math]
         └─ MixingInstanceSource(39aa7de): 20.3B tokens
            ├─ SamplingInstanceSource(cbc20a2): 2.0B tokens [megamath]
            │  └─ ConcatAndChunkInstanceSource(2b6a324): 3.9B tokens [megamath]
            │     └─ NumpyDocumentSource x 264: 3.9B tokens [megamath]
            └─ SamplingInstanceSource(857de5e): 18.3B tokens [dolminos2math]
               └─ ConcatAndChunkInstanceSource(b768b9a): 18.3B tokens [dolminos2math]
                  └─ NumpyDocumentSource x 415: 18.3B tokens [dolminos2math]

Curriculum learning
-------------------

The composable API also enables curriculum learning.
Suppose we want the first half of training to focus on 25% code + 75% math, and the second half
to focus on 75% code + 25% math.

We'll start by randomly splitting each of our sources, and since we'll want to set RNG seeds in
multiple places, we'll use the helper function :func:`set_composable_seed` to set the global starting seed
so that we don't have to set a different seed explicitly everywhere one is required::

   set_composable_seed(42)

   instance_sources = {
       "code_fim": make_instance_source("code_fim").random_split(0.25),
       "swallowcode": make_instance_source("swallowcode").random_split(0.25),
       "megamath": make_instance_source("megamath").random_split(0.75),
       "dolminos2math": make_instance_source("dolminos2math").random_split(0.75),
   }

And then we can create two separate mixes with the splits::

   def make_source_spec(label: str, split: int, ratio: float) -> MixingInstanceSourceSpecConfig:
       return MixingInstanceSource.Spec.Config(
           source=instance_sources[label][split],
           ratio=ratio,
           label=label,
       )
   
   mix_config1 = MixingInstanceSource.Config(
       source_specs=[
           MixingInstanceSource.Spec.Config(
               source=MixingInstanceSource.Config(
                   source_specs=[
                       make_source_spec("code_fim", 0, 0.5),
                       make_source_spec("swallowcode", 0, 0.5),
                   ]
               ),
               ratio=0.25,
               label="code",
           ),
           MixingInstanceSource.Spec.Config(
               source=MixingInstanceSource.Config(
                   source_specs=[
                       make_source_spec("megamath", 0, 0.1),
                       make_source_spec("dolminos2math", 0, 0.9),
                   ]
               ),
               ratio=0.75,
               label="math",
           ),
       ],
   )
   
   mix_config2 = MixingInstanceSource.Config(
       source_specs=[
           MixingInstanceSource.Spec.Config(
               source=MixingInstanceSource.Config(
                   source_specs=[
                       make_source_spec("code_fim", 1, 0.5),
                       make_source_spec("swallowcode", 1, 0.5),
                   ]
               ),
               ratio=0.75,
               label="code",
           ),
           MixingInstanceSource.Spec.Config(
               source=MixingInstanceSource.Config(
                   source_specs=[
                       make_source_spec("megamath", 1, 0.1),
                       make_source_spec("dolminos2math", 1, 0.9),
                   ]
               ),
               ratio=0.25,
               label="math",
           ),
       ],
   )
   
   mix1 = mix_config1.build("/tmp/dataset-common")
   mix1.visualize()
   mix2 = mix_config2.build("/tmp/dataset-common")
   mix2.visualize()

::

   MixingInstanceSource(6544ac1): 20.3B tokens
   ├─ SamplingInstanceSource(d28e959): 5.1B tokens [code]
   │  └─ MixingInstanceSource(08c8aa6): 9.4B tokens
   │     ├─ SamplingInstanceSource(c8e7179): 4.7B tokens [code_fim]
   │     │  └─ SlicedInstanceSource(02637e6): 5.3B tokens [code_fim]
   │     │     └─ ConcatAndChunkInstanceSource(adb4562): 21.4B tokens [code_fim]
   │     │        └─ NumpyDocumentSource x 474: 21.4B tokens [code_fim]
   │     └─ SamplingInstanceSource(520fff5): 4.7B tokens [swallowcode]
   │        └─ SlicedInstanceSource(fcaebbc): 4.7B tokens [swallowcode]
   │           └─ ConcatAndChunkInstanceSource(b2f2ef4): 18.8B tokens [swallowcode]
   │              └─ NumpyDocumentSource x 128: 18.8B tokens [swallowcode]
   └─ SamplingInstanceSource(b33b3f1): 15.2B tokens [math]
      └─ MixingInstanceSource(47322cf): 15.2B tokens
         ├─ SamplingInstanceSource(ccadff0): 1.5B tokens [megamath]
         │  └─ SlicedInstanceSource(c4cd38d): 2.9B tokens [megamath]
         │     └─ ConcatAndChunkInstanceSource(2b6a324): 3.9B tokens [megamath]
         │        └─ NumpyDocumentSource x 264: 3.9B tokens [megamath]
         └─ SamplingInstanceSource(476779e): 13.7B tokens [dolminos2math]
            └─ SlicedInstanceSource(75c18b6): 13.7B tokens [dolminos2math]
               └─ ConcatAndChunkInstanceSource(b768b9a): 18.3B tokens [dolminos2math]
                  └─ NumpyDocumentSource x 415: 18.3B tokens [dolminos2math]
      
   MixingInstanceSource(02f0d21): 20.3B tokens
   ├─ SamplingInstanceSource(76da23b): 15.2B tokens [code]
   │  └─ MixingInstanceSource(cb963a6): 28.3B tokens
   │     ├─ SamplingInstanceSource(5c8e643): 14.1B tokens [code_fim]
   │     │  └─ SlicedInstanceSource(f0ca032): 16.0B tokens [code_fim]
   │     │     └─ ConcatAndChunkInstanceSource(adb4562): 21.4B tokens [code_fim]
   │     │        └─ NumpyDocumentSource x 474: 21.4B tokens [code_fim]
   │     └─ SamplingInstanceSource(79187df): 14.1B tokens [swallowcode]
   │        └─ SlicedInstanceSource(0ae4650): 14.1B tokens [swallowcode]
   │           └─ ConcatAndChunkInstanceSource(b2f2ef4): 18.8B tokens [swallowcode]
   │              └─ NumpyDocumentSource x 128: 18.8B tokens [swallowcode]
   └─ SamplingInstanceSource(8e32820): 5.1B tokens [math]
      └─ MixingInstanceSource(25a8bb7): 5.1B tokens
         ├─ SamplingInstanceSource(70d1102): 507.6M tokens [megamath]
         │  └─ SlicedInstanceSource(d3ca4bc): 970.6M tokens [megamath]
         │     └─ ConcatAndChunkInstanceSource(2b6a324): 3.9B tokens [megamath]
         │        └─ NumpyDocumentSource x 264: 3.9B tokens [megamath]
         └─ SamplingInstanceSource(afd6a12): 4.6B tokens [dolminos2math]
            └─ SlicedInstanceSource(b53996f): 4.6B tokens [dolminos2math]
               └─ ConcatAndChunkInstanceSource(b768b9a): 18.3B tokens [dolminos2math]
                  └─ NumpyDocumentSource x 415: 18.3B tokens [dolminos2math]

When we build our :class:`ComposableDataLoader` we'll pass it both of those mixes, in order, and specify
the :class:`ShuffleStrategy` as :data:`~ShuffleStrategy.intra_source`
so that each mix is shuffled independently during its phase of training::

   data_loader = ComposableDataLoader.Config(
       tokenizer=tokenizer,
       global_batch_size=512 * sequence_length,
       shuffle_strategy=ShuffleStrategy.intra_source,
   ).build(mix1, mix2, work_dir="/tmp/dataloader-common")

Alternatively you could set ``sources_per_epoch=1`` to tell the data loader to use only the first source
for the first epoch, the second source for the second epoch, and so on::

   data_loader = ComposableDataLoader.Config(
       tokenizer=tokenizer,
       global_batch_size=512 * sequence_length,
       sources_per_epoch=1,
   ).build(mix1, mix2, work_dir="/tmp/dataloader-common")

Reference
---------
"""

from .concat_and_chunk_instance_source import (
    ConcatAndChunkInstanceSource,
    ConcatAndChunkInstanceSourceConfig,
)
from .data_loader import (
    ComposableDataLoader,
    ComposableDataLoaderConfig,
    InstanceFilterConfig,
    ShuffleStrategy,
)
from .instance_source import (
    ConcatenatedInstanceSource,
    ConcatenatedInstanceSourceConfig,
    Instance,
    InstanceSource,
    InstanceSourceConfig,
    SplitInstanceSourceConfig,
)
from .mixing_document_source import (
    MixingDocumentSource,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpec,
    MixingDocumentSourceSpecConfig,
)
from .mixing_instance_source import (
    MixingInstanceSource,
    MixingInstanceSourceConfig,
    MixingInstanceSourceSpec,
    MixingInstanceSourceSpecConfig,
)
from .mixing_token_source import (
    MixingTokenSource,
    MixingTokenSourceConfig,
    MixingTokenSourceSpec,
    MixingTokenSourceSpecConfig,
)
from .numpy_document_source import (
    NumpyDocumentSource,
    NumpyDocumentSourceConfig,
    NumpyDocumentSourceConfigBase,
    NumpyDocumentSourceMixConfig,
)
from .packing_instance_source import (
    LongDocStrategy,
    PackingInstanceSource,
    PackingInstanceSourceConfig,
)
from .random_instance_source import RandomInstanceSource, RandomInstanceSourceConfig
from .sampling_document_source import (
    SamplingDocumentSource,
    SamplingDocumentSourceConfig,
)
from .sampling_instance_source import (
    SamplingInstanceSource,
    SamplingInstanceSourceConfig,
)
from .sampling_token_source import SamplingTokenSource, SamplingTokenSourceConfig
from .sliced_instance_source import SlicedInstanceSource
from .sliced_token_source import SlicedTokenSource
from .source_abc import SourceABC
from .token_source import (
    ConcatenatedDocumentSource,
    ConcatenatedDocumentSourceConfig,
    ConcatenatedTokenSource,
    ConcatenatedTokenSourceConfig,
    DocumentSource,
    DocumentSourceConfig,
    InMemoryDocumentSource,
    InMemoryTokenSource,
    SplitTokenSourceConfig,
    TokenRange,
    TokenSource,
    TokenSourceConfig,
)
from .utils import reset_composable_seed, set_composable_seed

__all__ = [
    # Base classes.
    "SourceABC",
    "TokenSource",
    "TokenSourceConfig",
    "DocumentSource",
    "DocumentSourceConfig",
    "TokenRange",
    "InstanceSource",
    "InstanceSourceConfig",
    "Instance",
    "ComposableDataLoader",
    "ComposableDataLoaderConfig",
    # Token/document source implementations.
    "InMemoryTokenSource",
    "ConcatenatedTokenSource",
    "ConcatenatedTokenSourceConfig",
    "SlicedTokenSource",
    "SplitTokenSourceConfig",
    "SamplingTokenSource",
    "SamplingTokenSourceConfig",
    "MixingTokenSource",
    "MixingTokenSourceConfig",
    "InMemoryDocumentSource",
    "ConcatenatedDocumentSource",
    "ConcatenatedDocumentSourceConfig",
    "SamplingDocumentSource",
    "SamplingDocumentSourceConfig",
    "MixingDocumentSource",
    "MixingDocumentSourceConfig",
    "NumpyDocumentSource",
    "NumpyDocumentSourceConfigBase",
    "NumpyDocumentSourceConfig",
    "NumpyDocumentSourceMixConfig",
    # Instance source implementations.
    "ConcatAndChunkInstanceSource",
    "ConcatAndChunkInstanceSourceConfig",
    "PackingInstanceSource",
    "PackingInstanceSourceConfig",
    "ConcatenatedInstanceSource",
    "ConcatenatedInstanceSourceConfig",
    "SlicedInstanceSource",
    "SplitInstanceSourceConfig",
    "SamplingInstanceSource",
    "SamplingInstanceSourceConfig",
    "MixingInstanceSource",
    "MixingInstanceSourceConfig",
    "RandomInstanceSource",
    "RandomInstanceSourceConfig",
    # Other types.
    "InstanceFilterConfig",
    "LongDocStrategy",
    "ShuffleStrategy",
    "MixingInstanceSourceSpec",
    "MixingInstanceSourceSpecConfig",
    "MixingTokenSourceSpec",
    "MixingTokenSourceSpecConfig",
    "MixingDocumentSourceSpec",
    "MixingDocumentSourceSpecConfig",
    # Functions.
    "set_composable_seed",
    "reset_composable_seed",
]
