from typing import Any, List, Optional, Union

from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame

from oeciml.misc import retrieve_data
from oeciml.pipelines.documents import adicap


def _get_sampling_reception_date(
    df: sparkDataFrame,
    regex_sampling=r"((pr[ée]lev[ée]\sle)|(date\sd.\spr[ée]l[èe]vement))\s?:?\s{1,3}(\d{2}\/\d{2}\/\d{2,4})",
    regex_reception=r"((r?e[cç]u\sle)|(date\sde\sr.ception))\s?:?\s{1,3}(\d{2}\/\d{2}\/\d{2,4})",
) -> sparkDataFrame:
    df = df.withColumn("note_text_lower", F.lower(F.col("note_text")))

    df = df.withColumn(
        "reception",
        F.regexp_extract(F.col("note_text_lower"), pattern=regex_reception, idx=4),
    )

    df = df.withColumn(
        "reception_date", F.to_timestamp(F.col("reception"), "dd/MM/yyyy")
    )
    df = df.drop("reception")

    df = df.withColumn(
        "sampling",
        F.regexp_extract(F.col("note_text_lower"), pattern=regex_sampling, idx=4),
    )
    df = df.withColumn("sampling_date", F.to_timestamp(F.col("sampling"), "dd/MM/yyyy"))
    df = df.drop("sampling")

    df = df.drop("note_text_lower")

    return df


def get_biopsy_reports(
    documents_or_path: Union[sparkDataFrame, str],
    db: str,
    ghm_codes_of_stay: Optional[List[str]] = ["M", "Z", "K"],
    adicap_sampling_modes: Optional[List[str]] = [
        "BIOPSIE CHIRURGICALE",
        "CYTOPONCTION NON GUIDEE PAR IMAGERIE",
        "CYTOPONCTION GUIDEE PAR IMAGERIE",
        "HISTOPONCTION GUIDEE PAR IMAGERIE",
        "PONCTION BIOPSIE ET BIOPSIE INSTRUMENTALE NON GUIDEE PAR IMAGERIE",
        "BIOPSIE TRANSVASCULAIRE",
    ],
    return_cols: List[str] = [
        "note_id",
        "note_text",
        "note_datetime",
        "sampling_date",
        "reception_date",
        "note_class_source_value",
        "person_id",
        "visit_occurrence_id",
    ],
    **kwargs: Any,
) -> sparkDataFrame:
    if isinstance(documents_or_path, sparkDataFrame):
        docs = documents_or_path
    else:
        # Get documents previously selected
        spark, sql = retrieve_data.get_spark_sql()
        docs = spark.read.parquet(documents_or_path)

    # Pathological Reports
    pathological_reports = docs.filter(F.col("note_class_source_value") == "CR-ANAPATH")

    if ghm_codes_of_stay is not None:
        # Get GHM
        cost = retrieve_data.get_table("cost", db=db)

        # Keep only GHM codes of interest
        cost = cost.filter(F.col("letter_ghm").isin(list(ghm_codes_of_stay)))

        cost = cost.select(
            [
                "visit_occurrence_id",
            ]
        ).drop_duplicates()

        # Keep only reports of those stays
        pathological_reports = pathological_reports.join(
            cost,
            on="visit_occurrence_id",
            how="inner",
        )

    # Get ADICAP codes
    if adicap_sampling_modes is not None:
        adicap_codes = adicap.get_adicap_codes(pathological_reports)

        # Filter by ADICAP code
        docs_w_adicap_codes_not_chirurgy = adicap.filter_adicap(
            adicap_codes, adicap_sampling_modes=adicap_sampling_modes
        )

        pathological_reports = pathological_reports.join(
            docs_w_adicap_codes_not_chirurgy, on="note_id", how="inner"
        )

    # Extract dates
    pathological_reports = _get_sampling_reception_date(pathological_reports)

    # Return columns of `return_cols`
    pathological_reports = pathological_reports.select(return_cols)

    return pathological_reports
