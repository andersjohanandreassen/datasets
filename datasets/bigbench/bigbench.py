# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""TODO: Add a description here."""


import csv
import json
import numpy as np
import os

import datasets

from .bigbench_utils import _sanitize_task_data
from .bigbench_utils import make_nshot_dataset
from .bigbench_utils import default_format_fn


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to 
probe large language models, and extrapolate their future capabilities.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://github.com/google/BIG-bench"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "Apache License 2.0"

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLs = {
    'first_domain': "https://huggingface.co/great-new-dataset-first_domain.zip",
    'second_domain': "https://huggingface.co/great-new-dataset-second_domain.zip",
}


# class BigBenchConfig(datasets.BuilderConfig):
#     def __init__(self, url, lite, description, mode, *args, num_shots=1, **kwargs):
#         super().__init__(
#             *args,
#             # name=f"{type}-{task_no}",
#             **kwargs,
#         )
#         self.url = url
#         # self.type = type
#         # self.task_no = task_no
#         self.mode = 'multiple_choice'
#         self.num_shots = num_shots

class BigBenchConfig(datasets.BuilderConfig):
    def __init__(self,
                 task_name: str,
                 num_shots: int,
                 *args, 
                 subtask_name: str = None, 
                 input_prefix: str = None,
                 output_prefix: str = None,
                 choice_prefix: str = None, 
                 **kwargs):
        super().__init__(
            *args,
            # name=f"{type}-{task_no}",
            **kwargs,
        )
        self.task_name = task_name
        self.subtask_name = subtask_name
        self.num_shots = num_shots
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        self.choice_prefix = choice_prefix

        self.mode = 'multiple_choice'

        if self.subtask_name is None:
            self.url = f"https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/{self.task_name}/task.json"
        else:
            self.url = f"https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/{self.task_name}/{self.subtask_name}/task.json"


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class BigBench(datasets.GeneratorBasedBuilder):
    """The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to probe large language models, and extrapolate their future capabilities."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = BigBenchConfig


    def _info(self):
        if self.config.mode == 'multiple_choice':
            features = datasets.Features(
                {
                    "input": datasets.Value("string"),
                    "targets": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(datasets.Value("int32")),
                }
            )
        else: 
            features = datasets.Features(
                {
                    "input": datasets.Value("string"),
                    "target": datasets.Value("string"),
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        my_urls = self.config.url
        data_path = dl_manager.download(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_path,
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(
        self, filepath, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.
        rng = np.random.RandomState() #TODO(ajandreassen): make sure rng states are called in the same way as in big bench. Are they reused, or re-initialized?
        with open(filepath, encoding="utf-8") as f:
            task_data = json.load(f)
            
            if not self.config.input_prefix:
                input_prefix = task_data.get("example_input_prefix", "\nQ: ")
            else:
                input_prefix = self.config.input_prefix

            if not self.config.output_prefix:
                output_prefix = task_data.get("example_output_prefix", "\nA: ")
            else:
                output_prefix = self.config.output_prefix

            if not self.config.choice_prefix:
                choice_prefix = task_data.get("choice_prefix", "\n  choice: ")
            else:
                choice_prefix = self.config.choice_prefix

            examples = task_data['examples']
            examples = _sanitize_task_data(examples)
            examples = [default_format_fn(ex,
                                        input_prefix=input_prefix,
                                        output_prefix=output_prefix,
                                        choice_prefix=choice_prefix,
                                        rng = rng,
                                       ) for ex in examples ]
            examples = make_nshot_dataset(examples, shots=self.config.num_shots ,rng = rng)
            for id, example in enumerate(examples):
                if self.config.mode == "multiple_choice":
                    yield id, {
                        "input": example["input"],
                        "targets": list(example["target_scores"].keys()),
                        "labels": list(example["target_scores"].values()),
                    }
                else:
                    yield id, {
                        "input": example["input"],
                        "target": example["target"]
                    }
