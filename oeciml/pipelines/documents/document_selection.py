from typing import Any, List, Union

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame

from oeciml.misc import retrieve_data


def retrieve_docs_of_patient_set(
    patient_set_or_path: Union[List[str], sparkDataFrame, str],
    db: str,
    doc_types: List[str] = [
        "RCP",
        "CR-ANAPATH",
    ],
    cols_documents: List[str] = [
        "person_id",
        "visit_occurrence_id",
        "note_id",
        "note_datetime",
        "note_class_source_value",
        "note_text",
    ],
    **kwargs: Any,
):
    spark, sql = retrieve_data.get_spark_sql()
    # Patient set
    if isinstance(patient_set_or_path, list):
        patients_pd = pd.DataFrame(patient_set_or_path, columns=["person_id"])
        patients = spark.createDataFrame(patients_pd)
    elif isinstance(patient_set_or_path, sparkDataFrame):
        patients = patient_set_or_path
    else:
        patients = spark.read.parquet(patient_set_or_path)

    patients = patients.select(
        [
            "person_id",
        ]
    ).drop_duplicates()

    # Get table documents
    docs = retrieve_data.get_table(
        "documents",
        db=db,
        select_cols=cols_documents,
    )

    # Keep only documents of patient set
    docs_patients = docs.join(patients, how="inner", on="person_id")

    list_docs_data = []
    if "RCP" in doc_types:
        # Docs label RCP
        mdm_reports = docs_patients.where(F.col("note_class_source_value") == "RCP")

        # Docs label INCONNU with regex that match RCP
        unknown_documents = docs_patients.where(
            F.col("note_class_source_value") == "INCONNU"
        )

        unknown_documents = unknown_documents.withColumn(
            "title",
            F.regexp_extract(F.col("note_text"), pattern=r"(?i)intitul[ée].*", idx=0),
        )

        unknown_documents = unknown_documents.filter(
            F.lower(F.col("title")).rlike(r"rcp?")
        )
        unknown_documents = unknown_documents.drop(F.col("title"))
        unknown_documents = unknown_documents.replace(
            "INCONNU", "RCP", subset=["note_class_source_value"]
        )

        list_docs_data.append(mdm_reports)
        list_docs_data.append(unknown_documents)

    if "CR-ANAPATH" in doc_types:
        # Docs label CR-ANAPATH
        pathological_reports = docs_patients.where(
            F.col("note_class_source_value") == "CR-ANAPATH"
        )

        list_docs_data.append(pathological_reports)

    if len(list_docs_data) > 0:
        selected_docs = list_docs_data[0]
        for df in list_docs_data[1:]:
            selected_docs = selected_docs.union(df)

        # Filter out Null
        selected_docs = selected_docs.filter(F.col("note_text").isNotNull())

        # Filter out scanned documents
        pathological_reports = pathological_reports.filter(
            ~F.col("note_text").rlike("FICHIER PDF SCAN")
        )

        return selected_docs
