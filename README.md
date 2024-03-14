# Leveraging Information Redundancy of Real-World Data Through Distant Supervision
Accepted paper at LREC-COLING 2024

## SETUP

### Virtual environment
#### Create an environment
```bash
python -m venv .venv
```

#### Activate it
```bash
source .venv/bin/activate
```

#### Install packages
```bash
pip install pypandoc==1.7.5
pip install pyspark==2.4.8
poetry install
pip uninstall pypandoc
```

-----
# How to run the code ?
## Export environment variables and activate env
```bash
cd Ariel/oeci-ml
export ARROW_LIBHDFS_DIR=/usr/local/hadoop/usr/lib/
export HADOOP_HOME=/usr/local/hadoop
export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`
export PROJECT_LOCATION=Ariel/oeci-ml
conda deactivate
source .venv/bin/activate

```

## Run Scripts Data Generation
```bash
# Select the cohort
bash scripts/spark_submit.sh scripts/dataset_generation/cohort_selection.py --config configs/config_base.cfg

# Select clinical documents
bash scripts/spark_submit.sh scripts/dataset_generation/document_selection.py --config configs/config_base.cfg

# Classify documents as biopsy reports or not
bash scripts/spark_submit.sh scripts/dataset_generation/get_biopsy_reports.py --config configs/config_base.cfg
```

## Split data (train & test)
To split data into train dev test sets
```bash
# split between the test set and the others
python scripts/dataset_generation/split_set.py --config configs/config_base.cfg --split_set.stage="split_train_test"

# [alternatively] split between train and existing (annotated) data
python scripts/dataset_generation/split_set.py --config configs/config_base.cfg --split_set.stage="split_train_exisiting_data"
```

## Retrieve Surgery Procedure codes
```bash
bash scripts/spark_submit.sh scripts/dataset_generation/get_surgery_procedures.py --config configs/config_base.cfg
```

## NER dates
```bash
python scripts/dataset_generation/ner_dates.py --config configs/config_base.cfg
```

## Align train & dev dataset (programmatic annotation)
```bash
# align the others
python scripts/dataset_generation/date_alignement.py --config configs/config_PR_PS.cfg


# split train and dev depending on the alignment
python scripts/dataset_generation/split_set.py --config configs/config_base.cfg --split_set.stage="split_train_dev"
```

## Annotation
### << Annotation >>
> Notebooks/labeltool/biopsy_cytoponction
###  Read annotated set and post-process.
> notebooks/devtest/CreateValidationSet

## Export torch dataset
```bash
python scripts/dataset_generation/export_torch_dataset.py --config configs/config_PR_PS.cfg
```

## Models
### Cross entropy loss
```bash
sbatch scripts/train/torch/sbatch.sh configs/torch/ce/config_model_torch_ce.cfg
```

#### CE with dev set n_patients
```bash
sbatch scripts/train/torch/sbatch_split_dev.sh configs/torch/split_dev/config_model_torch_split_dev.cfg "--script.n_patients 10"
```

### Robust loss (ex. NCE & RCE)
```bash
sbatch scripts/train/torch/sbatch.sh configs/torch/nce_rce/config_model_torch_nce_rce_1.cfg
```

### O2U
```bash
# O2U
    ## Cyclical step
        ### train the model with a cyclical learning rate (o2u part 1)
        sbatch scripts/train/torch/sbatch_o2u.sh configs/torch/o2u/config_model_torch_o2u_cyclical_step.cfg

    ### Train with mask  (NCE-RCE)
        sbatch scripts/train/torch/sbatch.sh configs/torch/o2u/config_model_torch_o2u_train_NCERCE.cfg "--mask_params.forget_rate 0.3"
        sbatch scripts/train/torch/sbatch.sh configs/torch/o2u/config_model_torch_o2u_train_CE.cfg "--mask_params.forget_rate 0.3"

```
### LRT
```bash
# LRT
    # train the model and switch the labels according to a likelihood ratio test (AdaCorr or lrt)
    sbatch scripts/train/torch/sbatch_lrt.sh configs/torch/lrt/config_model_torch_lrt.cfg "--script.delta_base 1.3"
```

### Test
```bash
sbatch scripts/test/sbatch.sh configs/torch/config_test.cfg
```

### Test by biopsy status
```bash
sbatch scripts/test/sbatch_test_by_bp_status.sh configs/torch/config_test_by_bp_status.cfg
```
