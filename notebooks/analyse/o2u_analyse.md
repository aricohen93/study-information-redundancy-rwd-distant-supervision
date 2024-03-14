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
    display_name: oeci_ml_python
    language: python
    name: oeci_ml_python
---

```python
%load_ext autoreload
%autoreload 2
%config Completer.use_jedi = False
%load_ext jupyter_black
```

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as px
import plotly.graph_objects as go
```

```python
# pip install -U kaleido
```

```python
colors = px.colors.qualitative.Prism
```

```python
config_name = "config_base"
dataset_name = "PR_SP_PS"
dataset_name = "EL"
```

```python
output_path_loss_values = f"/export/home/cse/Ariel/oeci-ml/data/{config_name}/o2u/{dataset_name}/O2U_cyclical_step/loss_values.pt"
```

```python
# output_path_moving_loss = "/export/home/cse/Ariel/oeci-ml/data/config_all_cancer/o2u/moving_loss_O2U_cyclical_step.pt"
```

## loss distribution


# TODO
- loss rank per class
- bias de frases repetidas o pseudorepetidas con mal label

```python
loss_values = torch.load(output_path_loss_values)
```

```python
loss_values.shape
```

```python
# Calculate the number of data points and epochs
num_data = len(loss_values)
print("num_data:", num_data)
num_epochs = len(loss_values[0])
print("num_epochs:", num_epochs)
```

```python
# mean_normalized_loss_per_sample = torch.load(output_path_moving_loss)
```

## plot loss by group

```python
from itertools import zip_longest
```

```python
loss_values.mean(axis=1)
```

```python
loss_list = loss_values

# Step 1: Calculate the mean normalized loss for each sample across epochs
# mean_normalized_loss_per_sample = [np.mean(loss_value) for loss_value in loss_list]
mean_normalized_loss_per_sample = loss_values.mean(axis=1)

# Step 2: Determine the percentile boundaries for grouping the samples
sep = 10
percentiles = np.percentile(mean_normalized_loss_per_sample, range(0, 101, sep))

print("percentiles:", percentiles)

# Step 3: Group the samples based on their mean normalized loss values
grouped_samples = [[] for _ in range(len(percentiles) - 1)]

for i, loss in enumerate(mean_normalized_loss_per_sample):
    for j in range(len(percentiles) - 1):
        if percentiles[j] <= loss < percentiles[j + 1]:
            grouped_samples[j].append(i)
            break

groupedby10 = grouped_samples

# Step 4: Calculate the mean normalized loss values for each percentile group across epochs
mean_normalized_loss_per_group = []

for group in grouped_samples:
    group_losses = [loss_list[i] for i in group]
    # Use the zip function to group the elements by their positions
    transposed_lists = zip_longest(*group_losses, fillvalue=0)
    mean_losses_per_epoch = [np.mean(position) for position in transposed_lists]
    mean_losses_per_epoch = mean_losses_per_epoch[:20]
    mean_normalized_loss_per_group.append(mean_losses_per_epoch)

mean_normalized_loss_per_group10 = mean_normalized_loss_per_group

fig = go.Figure()

for i, mean_losses in enumerate(mean_normalized_loss_per_group):
    label = f"{i*sep}%-{(i+1)*sep}%"
    fig.add_trace(
        go.Scatter(
            x=list(range(len(mean_losses))),
            y=mean_losses,
            mode="lines",
            name=label,
            line=dict(color=colors[i]),
        )
    )

# Transpose the loss_list so that each sublist represents the loss values for each epoch
loss_list_transposed = zip_longest(*loss_list, fillvalue=0)

# Calculate the average loss value for each epoch
average_loss_per_epoch = [np.mean(loss_value) for loss_value in loss_list_transposed][
    :20
]

fig.add_trace(
    go.Scatter(
        x=list(range(len(average_loss_per_epoch))),
        y=average_loss_per_epoch,
        mode="lines",
        name="Average",
        line=dict(color="black", width=4),
    )
)

fig.update_layout(
    xaxis=dict(title="Epoch"),
    yaxis=dict(title="Mean Loss"),
    title="Mean Loss per Group",
    legend=dict(x=1, y=1),
    width=800,
    height=1200,
    showlegend=True,
)

# fig.update_yaxes(type="log")

# fig.write_image("../../data/1_1_pos_neg/config_base/o2u/biopsy_cytoponction/figures/loss_per_group_10epc_Adam_NoPretrained_val_is_train.png")
# fig.write_html("/export/home/cse/Ariel/oeci-ml/figures/o2u/loss_per_group.html")

fig.show(renderer="iframe")
```

```python
# mean_normalized_loss_per_group10
```

```python
# import pandas as pd
```

```python
# df = pd.DataFrame(mean_normalized_loss_per_group10)
# df.index.name = "group"
# df.reset_index(inplace=True)
# df = df.melt(
#     id_vars=["group"],
#     value_vars=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     value_name="mean_normalized_loss",
#     var_name="epoch",
# )
# df
```

```python
# import seaborn as sns
```

```python
# ax = sns.lineplot(
#     df,
#     x="epoch",
#     y="mean_normalized_loss",
#     hue="group",
#     palette=sns.color_palette("tab10"),
# )
# # ax.set(yscale="log")
# ax.set_ylim(bottom=0, top=1)
```

```python
loss_list = loss_values

# Step 1: Calculate the mean normalized loss for each sample across epochs
# mean_normalized_loss_per_sample = [np.mean(loss_value) for loss_value in loss_list]
mean_normalized_loss_per_sample = loss_values.mean(axis=1)

# Step 2: Determine the percentile boundaries for grouping the samples
sep = 2
percentiles = np.percentile(mean_normalized_loss_per_sample, range(0, 101, sep))

# Step 3: Group the samples based on their mean normalized loss values
grouped_samples = [[] for _ in range(len(percentiles) - 1)]

for i, loss in enumerate(mean_normalized_loss_per_sample):
    for j in range(len(percentiles) - 1):
        if percentiles[j] <= loss < percentiles[j + 1]:
            grouped_samples[j].append(i)
            break

# Step 4: Calculate the mean normalized loss values for each percentile group across epochs
mean_normalized_loss_per_group = []

for group in grouped_samples:
    group_losses = [loss_list[i] for i in group]
    # Use the zip function to group the elements by their positions
    transposed_lists = zip_longest(*group_losses, fillvalue=0)
    mean_losses_per_epoch = [np.mean(position) for position in transposed_lists]
    mean_losses_per_epoch = mean_losses_per_epoch[:20]
    mean_normalized_loss_per_group.append(mean_losses_per_epoch)

fig = go.Figure()

bottom50_transposed = zip_longest(*mean_normalized_loss_per_group[:40], fillvalue=0)
bottom50 = [np.mean(loss_value) for loss_value in bottom50_transposed]
fig.add_trace(
    go.Scatter(
        x=list(range(len(bottom50))),
        y=bottom50,
        mode="lines",
        name="0-80%",
        line=dict(color=colors[0]),
    )
)

for i, mean_losses in enumerate(mean_normalized_loss_per_group[40:50]):
    label = f" {(i+40)*sep}%-{(i+41)*sep}%"
    fig.add_trace(
        go.Scatter(
            x=list(range(len(mean_losses))),
            y=mean_losses,
            mode="lines",
            name=label,
            line=dict(color=colors[i + 1]),
        )
    )

# Transpose the loss_list so that each sublist represents the loss values for each epoch
loss_list_transposed = zip_longest(*loss_list, fillvalue=0)

# Calculate the average loss value for each epoch
average_loss_per_epoch = [np.mean(loss_value) for loss_value in loss_list_transposed][
    :20
]

fig.add_trace(
    go.Scatter(
        x=list(range(len(average_loss_per_epoch))),
        y=average_loss_per_epoch,
        mode="lines",
        name="Average",
        line=dict(color="black", width=4),
    )
)

fig.update_layout(
    xaxis=dict(title="Epoch"),
    yaxis=dict(title="Mean Loss"),
    title="Mean Loss per Group",
    legend=dict(x=1, y=1),
    width=800,
    height=1200,
    showlegend=True,
)

# fig.update_yaxes(type="log")

# fig.write_image("../data/1_1_pos_neg/config_base/o2u/biopsy&cytoponction/figures/loss_per_group_10epc_Adam_NoPretrained_groupedby2.png")
# fig.write_html("Alexandrine/oeci-ml/scripts/train/o2u_train/loss_per_group_10_6epc.html")

fig.show(renderer="iframe")
```

## text analysis

```python
import datasets
```

```python
from datasets import DatasetDict, Dataset
```

```python
import spacy
from spacy.util import filter_spans
from spacy import displacy
```

```python
data = datasets.load_from_disk(
    f"/data/scratch/cse/oeci-ml/data/{config_name}/dataset_config_{dataset_name}/"
)


data
```

```python
nlp = spacy.blank("eds")
```

```python
def visualisation_date_and_label(data, index):
    text = data["text"][index]
    start, end = data["span_start"][index], data["span_end"][index]
    label = data["label"][index]
    print(label)
    doc = nlp(text)
    doc.ents = [
        doc.char_span(start, end, label=str(label)),
    ]
    colors = {
        "False": "grey",
        "True": "green",
        "0": "grey",
        "1": "green",
        "2": "red",
    }
    options = {"ents": ["False", "True", "0", "1", "2"], "colors": colors}

    displacy.render(doc, style="ent", options=options)
```

```python
index = 0
```

```python
# data["train"][groupedby10[9]]["label"]
```

```python
len(groupedby10)
```

```python
# top 10 high loss
visualisation_date_and_label(Dataset.from_dict(data["train"][groupedby10[0]]), index)
print("####### \n index:", index)
index += 1
```

```python
index = 1500
```

```python
# top 30-40 high loss
visualisation_date_and_label(Dataset.from_dict(data['train'][groupedby10[1]]), index)
print('####### \n index:', index)
index +=1
```

```python
index = 0
```

```python
# bottom 10
visualisation_date_and_label(Dataset.from_dict(data["train"][groupedby10[0]]), index)
print("####### \n index:", index)
index += 1
```

```python
index = 105
```

```python
# bottom 10
visualisation_date_and_label(Dataset.from_dict(data["train"][groupedby10[6]]), index)
print("####### \n index:", index)
index += 1
```

## plot of True and False repartition

```python
repartition = [
    "0-10",
    "10-20",
    "20-30",
    "30-40",
    "40-50",
    "50-60",
    "60-70",
    "70-80",
    "80-90",
    "90-100",
]
```

```python
labels_groupedby10 = []
for group in groupedby10:
    labels = Dataset.from_dict(data["train"][group])["label"]
    labels_groupedby10.append(labels)
```

```python
# countlabels_groupedby10 = [[labels.count(True), labels.count(False)] for labels in labels_groupedby10]
```

```python
countlabels_groupedby10 = [
    [labels.count(0), labels.count(1), labels.count(2)] for labels in labels_groupedby10
]
```

```python
countlabels_groupedby10[0]
```

```python
fig = go.Figure()
x = repartition
fig.add_bar(
    x=x,
    y=np.array([label[0] for label in countlabels_groupedby10]) * 100 / 657,
    name="0",
    marker_color=colors[4],
)
fig.add_bar(
    x=x,
    y=np.array([label[1] for label in countlabels_groupedby10]) * 100 / 657,
    name="1",
    marker_color=colors[7],
)
fig.add_bar(
    x=x,
    y=np.array([label[2] for label in countlabels_groupedby10]) * 100 / 657,
    name="2",
    marker_color=colors[8],
)

fig.update_layout(barmode="relative")
# fig.write_image("../../data/1_1_pos_neg/config_base/o2u/biopsy_cytoponction/figures/repartition_per_group_10epc_Adam_NoPretrained.png")

fig.show(renderer="iframe")
```

## Plot variance

```python
len(mean_normalized_loss_per_group10)
```

```python
tranche = []
for i in range(len(mean_normalized_loss_per_group10)):
    tranche.append([repartition[i] for j in range(20)])

tranche_flat = []
for item in tranche:
    tranche_flat += item
```

```python
# delete first epochs
loss_per_group = [sublist for sublist in mean_normalized_loss_per_group10]
```

```python
normalized_loss_per_group = []
for i in range(len(loss_per_group)):
    normalized = np.array(loss_per_group[i]) - np.mean(loss_per_group[i])
    normalized_loss_per_group.append(normalized)
```

```python
normalized_loss_per_group_flat = (
    np.concatenate(normalized_loss_per_group).ravel().tolist()
)
```

```python
mean_normalized_loss_per_group_flat = []
for item in mean_normalized_loss_per_group10:
    mean_normalized_loss_per_group_flat += item
```

```python
len(tranche_flat)
```

```python
len(mean_normalized_loss_per_group_flat)
```

```python
df = pd.DataFrame(
    {
        "loss_value": mean_normalized_loss_per_group_flat,
        "normalized_loss_value": normalized_loss_per_group_flat,
        "tranche": tranche_flat,
    }
)
```

```python
import plotly.express as px

fig = px.box(df, x="tranche", y="normalized_loss_value", points=False)
fig.update_layout(
    xaxis=dict(title="Repartition (%)"),
    yaxis=dict(title="Normalized loss value"),
    width=800,
    height=600,
)
# fig.write_image("../../data/1_1_pos_neg/config_base/o2u/biopsy_cytoponction/figures/variance_per_group_10epc_Adam_NoPretrained.png")

fig.show(renderer="iframe")
```

```python
variances = [np.var(sublist) for sublist in normalized_loss_per_group]
```

```python
# fig = px.line(y=variances, line=dict(color=colors[6], width=4))
fig = go.Figure(
    go.Scatter(x=repartition, y=variances, line=dict(color=colors[6], width=3))
)
fig.update_layout(xaxis=dict(title="Repartition (%)"), yaxis=dict(title="Variance"))
fig.show(renderer="iframe")
```

```python

```
