---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: oeci_ml_local
    language: python
    name: oeci_ml_local
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
```

```python
import torch
```

## Create validation set

```python
# we kept 100 docs as a validation set in case, even if only 60 were annotated
# valid100 = pd.read_pickle("/export/home/cse/Alexandrine/oeci-ml/data/1_1_pos_neg/config_base/data/split_alignable/biopsy_cytoponction/valid_texts_config_base.pickle")
```

```python
# len(valid100)
```

```python
# valid100
```

```python
# valid_matched = valid100[valid100.alignment == 'dev matched']
```

```python
# valid_others = valid100[valid100.alignment == 'dev without cr']
```

```python
# # create a 60 docs validation set
# valid30_matched = valid_matched[:30]
# valid30_others = valid_others[:30]
```

```python
# valid_annotation_texts = pd.concat([valid30_matched, valid30_others])
```

```python
# valid_annotation_texts = valid_annotation_texts.sample(frac=1).reset_index(drop=True)
```

```python
# valid_annotation_texts
```

```python
# save validation texts
#valid_annotation_texts.to_pickle('../../data/config_base_pickle/split_alignable/biopsy_cytoponction/valid_annotation_texts_config_base.pickle')
```

## Read annotated set

```python
annotated = pd.read_pickle('/export/home/cse/Alexandrine/oeci-ml/notebooks/labeltool/biopsy_cytoponction/annotated.pickle')
```

```python
len(annotated)
```

```python
# drop what Emmanuelle manually added
annotated = annotated.drop(annotated[annotated['added_by_user'] == True].index)
```

```python
len(annotated)
```

```python
validation_set = annotated[['note_id', 'lexical_variant', 'offset_begin', 'offset_end', 'gold_label_value']]
```

```python
validation_set = validation_set.rename(columns={'offset_begin': 'text_start_char', 'offset_end':'text_end_char','gold_label_value':'label'})
```

```python
validation_set.insert(0, 'set', 'annotated_set')
```

```python
validation_set
```

```python
#validation_set.to_pickle('../../data/annotated_Emmanuelle/biopsy&cytoponction/annotated_set.pickle')
```

```python
#validation_set.to_csv('../../data/annotated_Emmanuelle/biopsy&cytoponction/annotated_set.csv')
```

## Extract dates

```python
from oeciml.pipelines.dataset_generation.dates.base import getDatesfromText
```

```python
valid_annotated = pd.read_csv('/export/home/cse/Alexandrine/oeci-ml/data/annotated_Emmanuelle/biopsy_cytoponction/annotated_set.csv')
```

```python
valid_annotated.head()
```

```python
len(valid_annotated)
```

```python
valid_annotated.label.value_counts()
```

```python
valid_annotated.note_id.nunique()
```

```python
valid_texts = pd.read_pickle('/export/home/cse/Alexandrine/oeci-ml/data/1_1_pos_neg/config_base/data/split_alignable/biopsy_cytoponction/valid_annotation_texts_config_base.pickle')
```

```python
len(valid_texts)
```

```python
valid_annotated.note_id = valid_annotated.note_id.astype(str)
valid_texts.note_id = valid_texts.note_id.astype(str)
```

```python
valid_texts = valid_texts[valid_texts.note_id.isin(valid_annotated.note_id)]
```

```python
len(valid_texts)
```

```python
n_before=30
n_after=45
ignore_excluded=True
drop_na = True
```

```python
dates_from_valid = getDatesfromText(
    n_before=n_before,
    n_after=n_after,
    ignore_excluded=ignore_excluded,
    drop_na = drop_na, # warning: this parameter appears in the name of the saved dataset
    attr="NORM",
)(valid_texts)
```

```python
dates_from_valid.head()
```

```python
len(dates_from_valid)
```

```python
#dates_from_valid.to_pickle('../../data/1_1_pos_neg/config_base/data/split_alignable/biopsy&cytoponction/valid_dropnaTrue/dates_from_valid.pickle')
```

```python
valid_labeled = dates_from_valid.merge(valid_annotated[["note_id", "text_start_char", "text_end_char", 'label']], how='inner', on=["note_id", "text_start_char", "text_end_char"], validate="one_to_one")
```

```python
valid_labeled.label.value_counts()
```

```python
len(valid_annotated), len(valid_labeled)
```

```python
valid_labeled.insert(0, 'set', 'dev')
```

```python
valid_labeled.insert(1, 'text_id', [i for i in range(len(valid_labeled))])
```

```python
valid_labeled.head()
```

```python
valid_labeled.query("lexical_variant =='11/08/2020'")
```

```python
#valid_labeled.to_pickle('../../data/1_1_pos_neg/config_base/data/split_alignable/biopsy&cytoponction/valid_dropnaTrue/valid_biopsy_dates_config_base.pickle')
```

# Replace date by mask

```python
from oeciml.misc.data_wrangling import wrap_replace_by_new_text
```

```python
from oeciml.pipelines.dataset_generation.dates.alignement import BaseDateTextAlignement
```

```python
valid_labeled = BaseDateTextAlignement.mask_entity(valid_labeled)
```

```python
valid_labeled.head(2)
```

```python
# valid_labeled.to_pickle(f"../../data/annotated/dev_set_biopsy_cytoponction-n_before={n_before}-n_after={n_after}-ignore_exluded={ignore_excluded}.pickle")
```

```python
print(valid_labeled.masked_span[1][129:135])
```

## Create dataset dict (train&val)

```python
from oeciml.torchmodules.data.dataset import TrainValidtoDatasetDict
```

```python
train = pd.read_pickle('/export/home/cse/Alexandrine/oeci-ml/data/1_1_pos_neg/config_base/data/split_alignable/biopsy_cytoponction/train_biopsy_dates_config_base.pickle')
```

```python
train = BaseDateTextAlignement.mask_entity(train)
```

```python
train.head()
```

```python
len(train)
```

```python
dataset_creator = TrainValidtoDatasetDict(
        text_col= "masked_span",
        id_col = "text_id",
        span_cols=  ["masked_span_start_char", "masked_span_end_char"],
        label_col = "label",
        # save_path = "/export/home/cse/Alexandrine/oeci-ml/data/1_1_pos_neg/config_base/final_dataset/biopsy_cytoponction_dataset/valid_dropnaTrueMasked/"
                             )

dataset = dataset_creator.generate(train_data=train,
        validation_data=valid_labeled,)
```

```python
dataset
```

```python
# import datasets
```

```python
# dataset = datasets.load_from_disk("/export/home/cse/Alexandrine/oeci-ml/data/1_1_pos_neg/config_base/final_dataset/biopsy_cytoponction_dataset/valid_dropnaTrueMasked/")
```

```python
# dataset["train"][0]
```

#  Split annotated into train / dev

```python
%load_ext autoreload
%autoreload 2
%config Completer.use_jedi = False
%load_ext jupyter_black
```

```python
import pandas as pd
```

```python
from oeciml.torchmodules.data.dataset import TrainValidtoDatasetDict
```

```python
valid_labeled = pd.read_pickle(
    "/export/home/cse/Ariel/oeci-ml/data/annotated/dev_set_biopsy_cytoponction-n_before=30-n_after=45-ignore_exluded=True.pickle"
)
```

```python
valid_labeled.head()
```

```python
n_patients = 60
```

```python
sample_valid_ids = valid_labeled.person_id.drop_duplicates().sample(n_patients)
```

```python
sample_valid = valid_labeled.loc[valid_labeled.person_id.isin(sample_valid_ids)]
```

```python
sample_valid.note_id.nunique()
```

```python
sample_valid.label.value_counts(normalize=True)
```

```python
valid_labeled.label.value_counts(normalize=True)
```

```python
n_entities = len(sample_valid)
n_entities
```

```python
# sample_valid.to_pickle(
#     f"/export/home/cse/Ariel/oeci-ml/data/annotated/valid_{n_patients}_patients_{n_entities}_entities.pickle"
# )
```

```python
f"/export/home/cse/Ariel/oeci-ml/data/annotated/valid_{n_patients}_patients_{n_entities}_entities.pickle"
```

```python
test_data = pd.read_pickle(
    "/export/home/cse/Ariel/oeci-ml/data/annotated/test_data.pickle"
)
```

```python
dataset_creator = TrainValidtoDatasetDict(
    text_col="masked_span",
    id_col="text_id",
    span_cols=["masked_span_start_char", "masked_span_end_char"],
    label_col="label",
    balance_to_class=-1,
    # save_path="/data/scratch/cse/oeci-ml/data/config_base/dataset_config_EL/",
)
```

```python

```

```python
valid_labeled.person_id.nunique()
```

```python
valid_labeled.note_id.nunique()
```

```python
len(valid_labeled)
```

```python
n_patients = 15
```

```python
train_ids = valid_labeled.person_id.drop_duplicates().sample(n_patients)
```

```python
n_entities = len(valid_labeled.loc[valid_labeled.person_id.isin(train_ids)])
n_entities
```

```python
dataset = dataset_creator.generate(
    train_data=valid_labeled.loc[valid_labeled.person_id.isin(train_ids)],
    validation_data=valid_labeled.loc[~valid_labeled.person_id.isin(train_ids)],
    test_data=test_data,
)
```

```python
len(dataset["train"])
```

```python
dataset
```

```python
valid_labeled = pd.read_pickle(
    "~/Ariel/oeci-ml/data/annotated/valid_15_patients_105_entities.pickle"
)
```

```python
dataset = dataset_creator.generate(
    train_data=valid_labeled,
)
```

```python
dataset
```

```python
dataset.save_to_disk("/data/scratch/cse/oeci-ml/data/config_base/dataset_dev/")
```

```python

```
