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
import pandas as pd
pd.set_option("max_columns",None)
```

We select the last RCP document by patient.

```python
raw_test_data = pd.read_pickle("../data/config_base/test_texts_config_base.pickle")
```

```python
raw_test_data.sort_values("note_datetime", inplace=True)
```

```python
test_data = raw_test_data.groupby("person_id").last()
```

```python
test_data.to_pickle("../data/config_base/test_last_rcp_config_base.pickle")
```

```python
test_data
```

```python
entities_path = "../data/config_base/test_dates_config_base.pickle"
```

```python
raw_entities = pd.read_pickle(entities_path)
```

```python
entities = raw_entities.merge(test_data[["note_id"]], on="note_id", how="inner")
```

```python
entities.note_id.nunique()
```

```python
entities.to_pickle("../data/config_base/test_dates_config_base_last_rcp.pickle")
```

```python

```

```python

```
