---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: alexandrine_kernel_local
    language: python
    name: alexandrine_kernel_local
---

```python tags=[]
%load_ext autoreload
%autoreload 2
```

```python tags=[]
import pandas as pd
```

## Import data

```python tags=[]
annotated = pd.read_pickle('../labeltool/biopsy_cytoponction/annotated.pickle')
```

```python tags=[]
annotated
```

```python tags=[]
annotated[annotated['added_by_user'] == True]
```

```python tags=[]
remarked = annotated[annotated.remark!=''][['gold_label_value', 'remark', 'span_start_char', 'span_end_char', 'span', 'note_id']]
```

```python tags=[]
remarked
```

## Visualisation

```python tags=[]
import spacy
from spacy.util import filter_spans
from spacy import displacy
```

```python tags=[]
nlp = spacy.blank("eds")
```

```python tags=[]
def visualisation_date_and_label(data, index, col_name):
    text = data.span[index]
    start, end = int(data['span_start_char'][index]), int(data['span_end_char'][index])
    label = data[col_name][index]
    doc = nlp(text)
    doc.ents = [doc.char_span(start, end, label=str(label)),]
    colors = {
    "False": "red",
    "True": "green",
    }
    options = {"ents": ["False", "True"], "colors": colors}

    displacy.render(doc, style="ent", options=options)
```

```python tags=[]
list_index = remarked.index
index = 0
```

```python tags=[]
visualisation_date_and_label(remarked, list_index[index], 'gold_label_value')
index += 1
```

```python tags=[]
validation_set = pd.read_pickle('../../data/annotated_Emmanuelle/biopsy_cytoponction/annotated_set.pickle')
```

```python tags=[]
validation_set.head(2)
```

# Qualité de l'alignement

```python tags=[]
valid = pd.read_pickle('../../data/1_1_pos_neg/config_base/data/split_alignable/biopsy_cytoponction/valid_annotation_texts_config_base.pickle')
```

```python tags=[]
len(valid)
```

```python tags=[]
valid_matched = valid[valid['alignment'] == 'dev matched']
```

```python tags=[]
from oeciml.misc.retrieve_data import arrowConnector
```

```python tags=[]
c = arrowConnector()
```

```python tags=[]
from oeciml.pipelines.dataset_generation.dates.alignement import DateTextAlignementV2 # rapport 1/10 pos/neg
```

```python tags=[]
generateV2 = DateTextAlignementV2()
```

```python tags=[]
biopsy_reports = c.get_pd_table('hdfs://bbsedsi/user/cse/oeciml/config_base/biopsy_reports.parquet')
```

```python tags=[]
len(biopsy_reports)
```

```python tags=[]
valid_aligned = generateV2.generate_dataset(biopsy_reports, valid, dates_to_match_col_name="sampling_date", col_name_dates_to_match_note_id='cr_anapath_id')
```

```python tags=[]
valid_aligned.head(2)
```

```python tags=[]
valid_aligned['label'].value_counts()
```

```python tags=[]
annotated_set = pd.read_pickle('../../data/annotated_Emmanuelle/biopsy_cytoponction/annotated_set.pickle')
```

```python tags=[]
annotated_set
```

```python tags=[]
neg, pos = annotated_set['label'].value_counts()
```

```python tags=[]
neg, pos
```

```python tags=[]
print('percentage positives:', pos/(pos+neg)*100, '%')
```

```python tags=[]
merged = annotated_set.merge(valid_aligned[["note_id", "text_start_char", "text_end_char", 'label']], how='inner', on=["note_id", "text_start_char", "text_end_char"], suffixes=('_valid_annotated', '_valid_aligned'))
```

```python tags=[]
len(merged)
```

```python tags=[]
nb_false_positives = len(merged[(merged['label_valid_aligned'] == True) & (merged['label_valid_annotated'] == False)])
nb_false_negatives = len(merged[(merged['label_valid_aligned'] == False) & (merged['label_valid_annotated'] == True)])

print("nb_false_positives :", nb_false_positives)
print("nb_false_negatives :", nb_false_negatives)
```

```python tags=[]
print('percentage false pos:', 7/30*100, '%, percentage false neg:', 13/300*100, '%')
```

```python tags=[]
merged
```

```python tags=[]
false_neg_df = merged[(merged['label_valid_aligned'] == False) & (merged['label_valid_annotated'] == True)]
```

```python tags=[]
false_neg_df = false_neg_df.merge(valid[['note_id', 'note_text']], how = 'inner', on=['note_id'])
```

```python tags=[]
false_pos_df = merged[(merged['label_valid_aligned'] == True) & (merged['label_valid_annotated'] == False)]
```

```python tags=[]
false_pos_df = false_pos_df.merge(valid[['note_id', 'note_text']], how = 'inner', on=['note_id'])
```

```python tags=[]
false_pos_df
```

```python tags=[]
def visualisation_date_and_label(data, index, col_name):
    text = data.note_text[index]
    start, end = data['text_start_char'][index], data['text_end_char'][index]
    label = data[col_name][index]
    doc = nlp(text)
    doc.ents = [doc.char_span(start, end, label=str(label)),]
    colors = {
    "False": "red",
    "True": "green",
    }
    options = {"ents": ["False", "True"], "colors": colors}

    displacy.render(doc, style="ent", options=options)
```

```python tags=[]
index = 0
```

```python tags=[]
visualisation_date_and_label(false_neg_df, index, 'label_valid_aligned')
index += 1
```

# Où le modèle se trompe-t-il ? Comparaison inférence vs annotation

```python tags=[]
inference_valid = pd.read_pickle("../../data/results_inference/results_inference_nce_rce_with_mask_20e_0.1fr.pickle")
```

```python tags=[]
inference_valid = inference_valid.rename(columns={'start': 'text_start_char', 'end':'text_end_char','biopsy':'label'})
```

```python tags=[]
#inference_valid
```

```python tags=[]
inference_vs_annotated = validation_set.merge(valid_aligned[["note_id", "text_start_char", "text_end_char", 'label']], how='inner', on=["note_id", "text_start_char", "text_end_char"], suffixes=('_valid_annotated', '_valid_inference'))
```

```python tags=[]
inference_vs_annotated = inference_vs_annotated.merge(valid[['note_id', 'note_text']], how = 'inner', on=['note_id'])
```

```python tags=[]
inference_vs_annotated
```

```python tags=[]
false_neg_inference = inference_vs_annotated[(inference_vs_annotated['label_valid_inference'] == False) & (inference_vs_annotated['label_valid_annotated'] == True)].reset_index()
```

```python tags=[]
false_neg_inference
```

```python tags=[]
false_pos_inference = inference_vs_annotated[(inference_vs_annotated['label_valid_inference'] == True) & (inference_vs_annotated['label_valid_annotated'] == False)].reset_index()
```

```python tags=[]
index = 0
```

```python tags=[]
visualisation_date_and_label(false_pos_inference, index, 'label_valid_inference')
index += 1
```
