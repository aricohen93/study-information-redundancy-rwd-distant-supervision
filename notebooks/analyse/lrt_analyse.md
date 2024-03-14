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
%config Completer.use_jedi = False %config Completer.use_jedi = False
```

```python tags=[]
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as px
import plotly.graph_objects as go
```

```python tags=[]
import datasets
```

```python tags=[]
colors = px.colors.qualitative.Prism
```

# new labels vs previous labels

```python tags=[]
labels = torch.load('../../data/1_1_pos_neg/config_base/lrt/biopsy_cytoponction/labels_lrt_test.pt')
```

```python tags=[]
print('num epochs:', len(labels))
print('num data:', len(labels[0]))
```

```python tags=[]
labels[0]==labels[1]
```

```python tags=[]
labels[0]==labels[6]
```

```python
new_train = torch.load('../../data/1_1_pos_neg/config_base/lrt/biopsy_cytoponction/new_data_train_lrt_test.pt')
```

```python tags=[]
new_train
```

```python tags=[]
new_labels = []
for example in new_train:
    label = int(example['label'].item())
    new_labels.append(label)
```

```python tags=[]
ancient_train = datasets.load_from_disk("../../data/1_1_pos_neg/config_base/final_dataset/biopsy_cytoponction_dataset/valid_dropnaTrue")['train']
```

```python tags=[]
ancient_train
```

```python tags=[]
ancient_labels = []
for example in ancient_train:
    label = int(example['label'])
    ancient_labels.append(label)
```

```python tags=[]
new_labels==ancient_labels
```

```python tags=[]
indexes_that_changed = []
for index, (first, second) in enumerate(zip(ancient_labels, new_labels)):
    if first != second:
        indexes_that_changed.append(index)
```

```python tags=[]
print('percentage of switched labels:', round(len(indexes_that_changed)/len(ancient_labels)*100, 2), '%')
```

## which data is changed?

```python tags=[]
import spacy
from spacy.util import filter_spans
from spacy import displacy
```

```python tags=[]
nlp = spacy.blank("eds")
```

```python tags=[]
def visualisation_date_and_label(data, index):
    text = data['text'][index]
    start, end = data['span_start'][index], data['span_end'][index]
    label = data['label'][index]
    doc = nlp(text)
    doc.ents = [doc.char_span(start, end, label=str(label)),]
    colors = {
    "False": "grey",
    "True": "green",
    }
    options = {"ents": ["False", "True"], "colors": colors}

    displacy.render(doc, style="ent", options=options)
```

```python tags=[]
index = 0
```

```python tags=[]
# visualisation de l'ancient label ; les labels affichés ont donc été modifiés par l'algorithme
visualisation_date_and_label(ancient_train, indexes_that_changed[index])
index += 1
```

# probabilities

```python tags=[]
probabilities = torch.load('../../data/1_1_pos_neg/config_base/lrt/biopsy_cytoponction/probabilities_ncerce_1.2d_noPretraining.pt')
```

```python tags=[]
print('num samples:', len(probabilities))
print('num classes:', len(probabilities[0]))
print('num epochs:', len(probabilities[0][0]))
```

## per sample

```python tags=[]
index = 0
```

```python tags=[]
sample_proba = probabilities[index]

trace1 = go.Scatter(x=list(range(1, 21)), y=sample_proba[0], mode='lines+markers', name='Class 1')
trace2 = go.Scatter(x=list(range(1, 21)), y=sample_proba[1], mode='lines+markers', name='Class 2')

data = [trace1, trace2]

layout = go.Layout(title='Probabilities by Epoch', xaxis=dict(title='Epoch'), yaxis=dict(title='Probability'))

fig = go.Figure(data=data, layout=layout)
fig.show()

index += 1
```

## averaged

```python tags=[]
averaged_probabilities = []
for class_probs in zip(*probabilities):
    averaged_class_probs = np.mean(class_probs, axis=0)
    averaged_probabilities.append(averaged_class_probs)

averaged_probabilities = np.array(averaged_probabilities, dtype=object)
```

```python tags=[]
trace1 = go.Scatter(x=list(range(1, 21)), y=averaged_probabilities[0], mode='lines+markers', name='Class 1')
trace2 = go.Scatter(x=list(range(1, 21)), y=averaged_probabilities[1], mode='lines+markers', name='Class 2')

data = [trace1, trace2]

layout = go.Layout(title='Averaged probabilities by Epoch', xaxis=dict(title='Epoch'), yaxis=dict(title='Probability'))

fig = go.Figure(data=data, layout=layout)
fig.show()
```

```python

```
