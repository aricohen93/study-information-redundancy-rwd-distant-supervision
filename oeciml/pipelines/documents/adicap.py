from pathlib import Path
from typing import Union

import spacy
from edsnlp.processing import pipe
from pandas import DataFrame as pdDataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.dataframe import DataFrame as sparkDataFrame

from oeciml.misc.retrieve_data import get_spark_sql

boolean_type = T._infer_type(True)
str_type = T._infer_type("")
int_type = T._infer_type(1)


def get_adicap_codes(
    documents_or_path: Union[sparkDataFrame, Path, pdDataFrame],
):
    if isinstance(documents_or_path, (sparkDataFrame, pdDataFrame)):
        docs = documents_or_path
    else:
        spark, sql = get_spark_sql()
        docs = spark.read.parquet(documents_or_path)
        # Anapath (Biopsy Documents)
        docs = docs.filter(F.col("note_class_source_value") == "CR-ANAPATH")

    # Define EDS-NLP pipeline
    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.adicap")

    # Apply pipeline
    codes = pipe(
        note=docs,
        nlp=nlp,
        additional_spans=["adicap"],
        extensions={
            "adicap.code": str_type,
            "adicap.sampling_mode": str_type,
            "adicap.technic": str_type,
            "adicap.organ": str_type,
            "adicap.pathology": str_type,
            "adicap.pathology_type": str_type,
            "adicap.behaviour_type": str_type,
        },
    )

    return codes


def filter_adicap(
    adicap_codes: sparkDataFrame,
    adicap_sampling_modes=[
        "BIOPSIE CHIRURGICALE",
        "CYTOPONCTION NON GUIDEE PAR IMAGERIE",
        "CYTOPONCTION GUIDEE PAR IMAGERIE",
        "HISTOPONCTION GUIDEE PAR IMAGERIE",
        "PONCTION BIOPSIE ET BIOPSIE INSTRUMENTALE NON GUIDEE PAR IMAGERIE",
        "BIOPSIE TRANSVASCULAIRE",
    ],
):
    # Filter in Notes where all sampling_mode that are in adicap_sampling_modes
    gp_df = adicap_codes.groupBy("note_id").agg(
        (F.min(F.col("adicap_sampling_mode").isin(list(adicap_sampling_modes)))).alias(
            "not_censure"
        )
    )

    gp_df = gp_df.filter(F.col("not_censure"))
    gp_df = gp_df.drop("not_censure")

    return gp_df
