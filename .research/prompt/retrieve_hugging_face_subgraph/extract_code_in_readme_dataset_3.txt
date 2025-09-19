
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
annotations_creators:
- expert-generated
- machine-generated
language_creators:
- expert-generated
language:
- en
license:
- mit
multilinguality:
- monolingual
size_categories:
- 100K<n<1M
- 10K<n<100K
- 1K<n<10K
source_datasets:
- original
task_categories:
- question-answering
task_ids:
- multiple-choice-qa
paperswithcode_id: pubmedqa
pretty_name: PubMedQA
config_names:
- pqa_artificial
- pqa_labeled
- pqa_unlabeled
dataset_info:
- config_name: pqa_artificial
  features:
  - name: pubid
    dtype: int32
  - name: question
    dtype: string
  - name: context
    sequence:
    - name: contexts
      dtype: string
    - name: labels
      dtype: string
    - name: meshes
      dtype: string
  - name: long_answer
    dtype: string
  - name: final_decision
    dtype: string
  splits:
  - name: train
    num_bytes: 443501057
    num_examples: 211269
  download_size: 233411194
  dataset_size: 443501057
- config_name: pqa_labeled
  features:
  - name: pubid
    dtype: int32
  - name: question
    dtype: string
  - name: context
    sequence:
    - name: contexts
      dtype: string
    - name: labels
      dtype: string
    - name: meshes
      dtype: string
    - name: reasoning_required_pred
      dtype: string
    - name: reasoning_free_pred
      dtype: string
  - name: long_answer
    dtype: string
  - name: final_decision
    dtype: string
  splits:
  - name: train
    num_bytes: 2088898
    num_examples: 1000
  download_size: 1075513
  dataset_size: 2088898
- config_name: pqa_unlabeled
  features:
  - name: pubid
    dtype: int32
  - name: question
    dtype: string
  - name: context
    sequence:
    - name: contexts
      dtype: string
    - name: labels
      dtype: string
    - name: meshes
      dtype: string
  - name: long_answer
    dtype: string
  splits:
  - name: train
    num_bytes: 125922964
    num_examples: 61249
  download_size: 66010017
  dataset_size: 125922964
configs:
- config_name: pqa_artificial
  data_files:
  - split: train
    path: pqa_artificial/train-*
- config_name: pqa_labeled
  data_files:
  - split: train
    path: pqa_labeled/train-*
- config_name: pqa_unlabeled
  data_files:
  - split: train
    path: pqa_unlabeled/train-*
---

# Dataset Card for [Dataset Name]

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** [PubMedQA homepage](https://pubmedqa.github.io/ )
- **Repository:** [PubMedQA repository](https://github.com/pubmedqa/pubmedqa)
- **Paper:** [PubMedQA: A Dataset for Biomedical Research Question Answering](https://arxiv.org/abs/1909.06146)
- **Leaderboard:**  [PubMedQA: Leaderboard](https://pubmedqa.github.io/)

### Dataset Summary

The task of PubMedQA is to answer research questions with yes/no/maybe (e.g.: Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?) using the corresponding abstracts.

### Supported Tasks and Leaderboards

The official leaderboard is available at: https://pubmedqa.github.io/.

500 questions in the `pqa_labeled` are used as the test set. They can be found at https://github.com/pubmedqa/pubmedqa.

### Languages

English

## Dataset Structure

### Data Instances

[More Information Needed]

### Data Fields

[More Information Needed]

### Data Splits

[More Information Needed]

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

[More Information Needed]

### Citation Information

[More Information Needed]

### Contributions

Thanks to [@tuner007](https://github.com/tuner007) for adding this dataset.
Output:
{
    "extracted_code": ""
}
