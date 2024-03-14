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
%config Completer.use_jedi = False
%load_ext jupyter_black

```

```python
import pandas as pd
import numpy as np
```

```python
import torch
```

## Read annotated set

```python
annotated = pd.read_pickle(
    "/export/home/cse/Ariel/oeci-ml/data/test_data/config_base/annotated.pickle"
)
```

```python
len(annotated)
```

```python
# drop what Emmanuelle manually added
annotated = annotated.drop(annotated[annotated["added_by_user"] == True].index)
```

```python
len(annotated)
```

```python
validation_set = annotated[
    ["note_id", "lexical_variant", "offset_begin", "offset_end", "gold_label_value"]
]
```

```python
validation_set = validation_set.rename(
    columns={
        "offset_begin": "text_start_char",
        "offset_end": "text_end_char",
        "gold_label_value": "label",
    }
)
```

```python
validation_set.insert(0, "set", "annotated_set")
```

```python
validation_set
```

## Extract dates

```python
from oeciml.pipelines.dataset_generation.dates.base import getDatesfromText
```

```python
from oeciml.misc.retrieve_data import get_table
from pyspark.sql import functions as F
```

```python
valid_texts = pd.read_pickle(
    "/export/home/cse/Ariel/oeci-ml/data/test_data/config_base/texts.pickle"
)
```

```python
valid_texts.head()
```

```python
len(valid_texts)
```

```python
validation_set.note_id = validation_set.note_id.astype(str)
valid_texts.note_id = valid_texts.note_id.astype(str)
```

```python
valid_texts = valid_texts[valid_texts.note_id.isin(validation_set.note_id)]
```

```python
valid_texts = valid_texts.query("vu")
```

```python
len(valid_texts)
```

```python
df_note_datetime = get_table(db="cse__20220502", table="documents")
```

```python
df_note_datetime = df_note_datetime.select(["person_id", "note_id", "note_datetime"])
```

```python
df_note_datetime = df_note_datetime.filter(
    F.col("note_id").isin(valid_texts.note_id.to_list())
)
```

```python
# df_note_datetime = df_note_datetime.toPandas()
```

```python
# df_note_datetime.to_pickle(
#     "/export/home/cse/Ariel/oeci-ml/data/test_data/config_base/note_datetime.pickle"
# )
```

```python
df_note_datetime = pd.read_pickle(
    "/export/home/cse/Ariel/oeci-ml/data/test_data/config_base/note_datetime.pickle"
)
```

```python
valid_texts = valid_texts.merge(
    df_note_datetime, on="note_id", how="inner", validate="one_to_one"
)
```

```python
n_before = 30
n_after = 45
ignore_excluded = True
drop_na = True
```

```python
dates_from_valid = getDatesfromText(
    n_before=n_before,
    n_after=n_after,
    ignore_excluded=ignore_excluded,
    drop_na=drop_na,  # warning: this parameter appears in the name of the saved dataset
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
# dates_from_valid.to_pickle('../../data/1_1_pos_neg/config_base/data/split_alignable/biopsy&cytoponction/valid_dropnaTrue/dates_from_valid.pickle')
```

```python
valid_labeled = dates_from_valid.merge(
    validation_set[["note_id", "text_start_char", "text_end_char", "label"]],
    how="inner",
    on=["note_id", "text_start_char", "text_end_char"],
    validate="one_to_one",
)
```

```python
valid_labeled.label.value_counts()
```

```python
len(validation_set), len(valid_labeled)
```

```python
validation_set.note_id.nunique()
```

```python
valid_labeled.insert(0, "set", "test")
```

```python
valid_labeled.insert(1, "text_id", [i for i in range(len(valid_labeled))])
```

```python
valid_labeled.head()
```

```python
valid_labeled.note_id.nunique()
```

```python
print(valid_labeled.query("label").span.iloc[0])
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
print(valid_labeled.masked_span[0][75:81])
```

```python
# valid_labeled.to_pickle(
#     "/export/home/cse/Ariel/oeci-ml/data/annotated/test_data.pickle"
# )
```

## Create dataset dict (train&val)

```python
from oeciml.torchmodules.data.dataset import TrainValidtoDatasetDict
```

```python
test_data = pd.read_pickle(
    "/export/home/cse/Ariel/oeci-ml/data/annotated/test_data.pickle"
)
```

```python
len(test_data)
```

```python
test_data.note_id.nunique()
```

```python
test_data.head()
```

```python
test_data.label.value_counts()
```

```python
from confit import Config
```

```python
conf = Config.from_disk("../../configs/config_base.cfg", resolve=True)
```

```python
biopsy_reports_path = conf["get_biopsy_reports"]["output_path"]
```

```python
biopsy_reports_path = biopsy_reports_path.format(
    user="cse",
    project_name=conf["project"]["project_name"],
    conf_name="config_base",
)
biopsy_reports_path
```

```python
from oeciml.misc.utils import file_reader
```

```python
biopsy_reports = file_reader(biopsy_reports_path)
```

```python
biopsy_reports["has_bp_report"] = True
```

```python
biopsy_reports = biopsy_reports.groupby("person_id", as_index=False).agg(
    has_bp_report=("has_bp_report", "first")
)
```

```python
test_data = test_data.merge(
    biopsy_reports, on="person_id", how="left", validate="many_to_one"
)
```

```python
test_data["has_bp_report"].fillna(False, inplace=True)
```

```python
test_data.drop_duplicates("person_id").has_bp_report.value_counts()
```

```python
test_data.groupby("has_bp_report").label.value_counts()
```

```python
# test_data.to_pickle(
#     "/export/home/cse/Ariel/oeci-ml/data/annotated/test_data.pickle"
# )
```

```python
# train = pd.read_pickle('/export/home/cse/Alexandrine/oeci-ml/data/1_1_pos_neg/config_base/data/split_alignable/biopsy_cytoponction/train_biopsy_dates_config_base.pickle')
```

```python
# train = BaseDateTextAlignement.mask_entity(train)
```

```python
# train.head()
```

```python
# len(train)
```

```python
dataset_creator = TrainValidtoDatasetDict(
    text_col="masked_span",
    id_col="text_id",
    span_cols=["masked_span_start_char", "masked_span_end_char"],
    label_col="label",
    # save_path = "/export/home/cse/Alexandrine/oeci-ml/data/1_1_pos_neg/config_base/final_dataset/biopsy_cytoponction_dataset/valid_dropnaTrueMasked/"
)

dataset = dataset_creator.generate(
    test_data=test_data.query("has_bp_report"),
    validation_data=test_data.query("~has_bp_report"),
)
```

```python
ds_w_bp = dataset.pop("test")
ds_wo_bp = dataset.pop("val")
```

```python
dataset["test_with_bp"] = ds_w_bp
dataset["test_without_bp"] = ds_wo_bp
```

```python
dataset
```

```python
# dataset.save_to_disk(
#     f"/data/scratch/cse/oeci-ml/data/config_base/dataset_test_by_bp_status/"
# )
```

```python
dataset_creator = TrainValidtoDatasetDict(
    text_col="masked_span",
    id_col="text_id",
    span_cols=["masked_span_start_char", "masked_span_end_char"],
    label_col="label",
    # save_path = "/export/home/cse/Alexandrine/oeci-ml/data/1_1_pos_neg/config_base/final_dataset/biopsy_cytoponction_dataset/valid_dropnaTrueMasked/"
)

dataset = dataset_creator.generate(test_data=test_data)
```

```python
dataset
```

```python
# dataset.save_to_disk(f"/data/scratch/cse/oeci-ml/data/config_base/dataset_test/")
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
dataset_creator = TrainValidtoDatasetDict(
        text_col= "masked_span",
        id_col = "text_id",
        span_cols=  ["masked_span_start_char", "masked_span_end_char"],
        label_col = "label",
        balance_to_class = -1,
        # save_path = "/export/home/cse/Alexandrine/oeci-ml/data/1_1_pos_neg/config_base/final_dataset/biopsy_cytoponction_dataset/valid_dropnaTrueMasked/"
                             )
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
n_patients = 40
```

```python
train_ids = valid_labeled.person_id.drop_duplicates().sample(n_patients)
```

```python
n_entities = len(valid_labeled.loc[valid_labeled.person_id.isin(train_ids)])
n_entities
```

```python
dataset = dataset_creator.generate(train_data=valid_labeled.loc[valid_labeled.person_id.isin(train_ids)],
        validation_data=valid_labeled.loc[~valid_labeled.person_id.isin(train_ids)],)
```

```python
dataset
```

```python
dataset.save_to_disk(
    f"/data/scratch/cse/oeci-ml/data/config_base/dataset_annotated_{n_patients}_patients_{n_entities}_entities/"
)
```

```python
f"/data/scratch/cse/oeci-ml/data/config_base/dataset_annotated_{n_patients}_patients_{n_entities}_entities/"
```
