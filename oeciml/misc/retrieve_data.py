import os
from typing import Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import parquet
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame

from oeciml.misc.constants import dict_hospitals, i2b2_renaming, table_mapping


# Create Spark Session
def get_spark_sql():
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    sql = spark.sql
    return spark, sql


def get_table(
    table: str,
    db: str,
    db_type: str = "i2b2",
    rename_cols: Optional[Dict[str, str]] = None,
    select_cols: Optional[List[str]] = None,
    **kwargs,
) -> sparkDataFrame:
    if db_type == "i2b2":
        if db == "edsprod":
            tables = table_mapping[db_type][db]
        else:
            tables = table_mapping[db_type]["other"]

        query_select = ",".join(
            ["{k} AS {v}".format(k=k, v=v) for k, v in i2b2_renaming[table].items()]
        )

    elif db_type == "omop":
        tables = table_mapping[db_type]["all"]

        query_select = ",".join([f"{v}" for v in i2b2_renaming[table].values()])

    else:
        raise ValueError(
            f"`db_type` should be one of ('i2b2', 'omop') and not {db_type}"
        )

    # Query table
    table_name = tables[table]
    spark, sql = get_spark_sql()
    df = sql(f"""SELECT {query_select} FROM {db}.{table_name}""")

    # Special maping for i2b2 :
    if db_type == "i2b2":
        # icd10
        if table == "icd10":
            df = df.withColumn(
                "condition_source_value",
                F.substring(F.col("condition_source_value"), 7, 20),
            )

        # CCAM
        elif table == "procedure":
            df = df.withColumn(
                "procedure_source_value",
                F.substring(F.col("procedure_source_value"), 6, 20),
            )

        # GHM
        elif table == "cost":
            df = df.withColumn(
                "drg_source_value", F.substring(F.col("concept_cd"), 5, 20)
            )
            df = df.withColumn("letter_ghm", F.substring(F.col("concept_cd"), 7, 1))
            df = df.drop("concept_cd")

        # Documents
        elif table == "documents":
            df = df.withColumn(
                "note_class_source_value",
                F.substring(F.col("note_class_source_value"), 4, 100),
            )

            df = df.drop_duplicates(subset=["note_id"])

        # Hospital codes
        elif table == "care_site":
            df = df.withColumn(
                "care_site_source_value",
                F.substring(F.col("care_site_source_value"), 5, 3),
            )
            df = df.dropDuplicates()

            key_hospitals_pd = pd.DataFrame.from_dict(dict_hospitals)
            key_hospitals_pd.rename_axis(index="care_site_source_value", inplace=True)
            key_hospitals_pd.reset_index(inplace=True, drop=False)

            key_hospitals = spark.createDataFrame(
                key_hospitals_pd[
                    ["care_site_source_value", "care_site_name", "cd_long_gh"]
                ]
            )
            df = df.join(
                key_hospitals.hint("broadcast"), how="left", on="care_site_source_value"
            )
            df = df.na.fill("INCONNU", subset=["care_site_name", "cd_long_gh"])

        # UFR
        elif table == "visit_detail":
            df = df.withColumn(
                "care_site_id", F.substring(F.col("care_site_id"), 5, 20)
            )
        # biology
        elif table == "biology":
            df = df.withColumn(
                "biology_source_value",
                F.substring(F.col("biology_source_value"), 5, 20),
            )

    if rename_cols:
        for k, v in rename_cols.items():
            df = df.withColumnRenamed(k, v)

    if select_cols:
        df = df.select(select_cols)
    return df


def retrieve_codes_icd(
    icd10_values: Optional[Union[str, List[str]]],
    db: str,
    db_type: str,
    claim_data_type: Optional[str],
    diagnostic_types: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> sparkDataFrame:
    """
    Retrieve a sparkDataFrame with visits that fulfill the specified ICD10 diagnostics.

    Parameters
    ----------
    bloc_letter: str,
        the chapter of the ICD-10 code to search. For example, if we want the codes between X60 and X84, so bloc_letter == "X"
    bloc_number_min: int,
        the first two numerical digits of the code
    claim_data_type: str,
        One of {'ORBIS','AREM', 'HEGP', None}


    Returns
    -------
    Returns a spark.DataFrame with the visits and ICD10 codes that fulfill the conditions. Visits could be duplicated if they have multiple codes.

    """

    # get table
    icd10 = get_table(table="icd10", db=db, db_type=db_type)

    # Keep only claim_data_type
    if claim_data_type:
        icd10 = icd10.where(F.col("cdm_source") == claim_data_type)

    if diagnostic_types:
        if isinstance(diagnostic_types, str):
            icd10 = icd10.where(
                F.col("condition_status_source_value") == diagnostic_types
            )
        elif isinstance(diagnostic_types, (list, tuple)):
            icd10 = icd10.where(
                F.col("condition_status_source_value").isin(list(diagnostic_types))
            )
        else:
            raise TypeError("diagnostic_types should be a str or list")

    # Extract codes from column
    # Make columns with the ICD10 code with 2 and 3 digits. Example: C40 & C401
    icd10 = icd10.withColumn(
        "condition_source_value_short_2", F.substring("condition_source_value", 1, 3)
    )
    icd10 = icd10.withColumn(
        "condition_source_value_short_3", F.substring("condition_source_value", 1, 4)
    )

    # Filter by codes
    if icd10_values:
        if isinstance(icd10_values, str):
            icd10 = icd10.where(
                (F.col("condition_source_value_short_2") == icd10_values)
                | (F.col("condition_source_value_short_3") == icd10_values)
            )
        elif isinstance(icd10_values, (list, tuple)):
            icd10 = icd10.where(
                (F.col("condition_source_value_short_2").isin(list(icd10_values)))
                | (F.col("condition_source_value_short_3").isin(list(icd10_values)))
            )
        else:
            raise TypeError("icd_values should be a str or list")

    return icd10


class arrowConnector:
    def __init__(self, path_table=None, db=None, table=None):
        self.path_table = path_table
        self.db = db
        if db and table:
            spark, sql = get_spark_sql()
            self.path_table = (
                sql(f"desc formatted {db}.{table}")
                .filter("col_name=='Location'")
                .collect()[0]
                .data_type
            )
            self.db = os.path.dirname(self.path_table)

        if path_table:
            self.path_table = path_table

        if (db) and (table is None):
            self.db = db

    def get_pd_fragment(
        self,
        path_table=None,
        table_name=None,
        types_mapper=None,
        integer_object_nulls=True,
        date_as_object=False,
    ):
        if path_table:
            self.path_table = path_table

        if table_name:
            self.path_table = os.path.join(self.db, table_name)

        # Import the parquet as ParquetDataset
        parquet_ds = parquet.ParquetDataset(
            self.path_table, use_legacy_dataset=False
        )

        # Partitions of ds
        fragments = iter(parquet_ds.fragments)

        # Set initial length
        length = 0

        # One partition
        while length < 1:
            fragment = next(fragments)

            # pyarrow.table of a fragment
            table = fragment.to_table()

            length = len(table)

        # Import to pandas the fragment
        table_pd = table.to_pandas(
            types_mapper=types_mapper,
            integer_object_nulls=integer_object_nulls,
            date_as_object=date_as_object,
        )
        return table_pd

    def count_fragments_length(self, path_table=None, table_name=None):
        if path_table:
            self.path_table = path_table

        if table_name:
            self.path_table = os.path.join(self.db, table_name)
        # Import the parquet as ParquetDataset
        parquet_ds = parquet.ParquetDataset(
            self.path_table, use_legacy_dataset=False
        )

        # Partitions of ds
        fragments = iter(parquet_ds.fragments)

        lengths = []
        for fragment in fragments:
            # pyarrow.table of a fragment
            table = fragment.to_table()
            lengths.append(len(table))

        return lengths

    def get_pd_table(
        self,
        path_table=None,
        table_name=None,
        types_mapper=None,
        integer_object_nulls=True,
        date_as_object=False,
        filter_values_keep=None,
        filter_values_avoid=None,
        cast_to_tz: Optional[str] = None,
        filter_col="person_id",
        columns = None,
    ):
        if path_table:
            self.path_table = path_table

        if table_name:
            self.path_table = os.path.join(self.db, table_name)

        filters = None
        if (filter_values_keep or filter_values_avoid) is not None:
            filters = []
            if filter_values_keep:
                filters.append((filter_col, "in", filter_values_keep))
            if filter_values_avoid:
                filters.append((filter_col, "not in", filter_values_avoid))
           

        table = parquet.read_table(self.path_table, filters=filters, columns=columns)

        df = table.to_pandas(
            date_as_object=date_as_object,
            types_mapper=types_mapper,
            integer_object_nulls=integer_object_nulls,
        )

        if cast_to_tz is not None:
            df = self.cast_to_tz(df, tz=cast_to_tz)
        return df

    @staticmethod
    def cast_to_tz(df, tz="Europe/Paris"):
        cols = df.select_dtypes(include=["datetime64"]).columns
        for col in cols:
            df[col] = df[col].dt.tz_localize("UTC")

            df[col] = df[col].dt.tz_convert(tz)
        return df
