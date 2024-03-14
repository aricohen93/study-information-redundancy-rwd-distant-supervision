from typing import Any, Optional

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame

from oeciml.misc.retrieve_data import get_spark_sql, get_table
from oeciml.registry import registry


@registry.structured_dates("ProcedureSelector")
class ProcedureSelector:
    def __init__(self, db: str, path_codes: str, tz: str = "Europe/Paris") -> None:
        self.db = db
        self.path_codes = path_codes
        self.tz = tz

    @staticmethod
    def read_codes(path_codes: str) -> pd.DataFrame:
        """
        Return a pd.DataFrame with columns `Code` et `Libelle`
        """
        raise NotImplementedError

    def filter_by_visit(
        self, procedures: sparkDataFrame, path_filter_by_visit: str
    ) -> sparkDataFrame:
        return procedures

    def __call__(self, path_filter_by_visit: Optional[str] = None) -> Any:
        # Get spark session
        spark, sql = get_spark_sql()

        # All procedures (CCAM)
        procedures = get_table(
            "procedure",
            db=self.db,
        )

        # Filter by visit
        if path_filter_by_visit is not None:
            procedures = self.filter_by_visit(
                procedures, path_filter_by_visit=path_filter_by_visit
            )

        # Read Procedure codes
        codes = self.read_codes(self.path_codes)

        # Cast these codes to spark
        codes_spark = spark.createDataFrame(codes)

        # Get procedures of these codes )
        procedures = procedures.join(
            codes_spark.hint("broadcast"),
            on=((codes_spark.Code == procedures.procedure_source_value)),
            how="inner",
        )

        # Groupby visit
        procedures_agg = procedures.groupby(["visit_occurrence_id", "person_id"]).agg(
            F.min(F.col("procedure_start_datetime")).alias("procedure_date")
        )

        # Cast to TZ
        procedures_agg = procedures_agg.withColumn(
            "procedure_date",
            F.from_utc_timestamp(F.col("procedure_date"), self.tz),
        )

        return procedures_agg


@registry.structured_dates("SurgeryProcedures")
class SurgeryProcedures(ProcedureSelector):
    def __init__(self, db: str, path_codes: str) -> None:
        super().__init__(db, path_codes)

    @staticmethod
    def read_codes(path_codes: str) -> pd.DataFrame:
        # Read codes exerese
        ccam = pd.read_excel(path_codes)

        idx_exerese = (
            ccam[
                [
                    "Exerese_abdo_pelvien",
                    "Exerese_ORL",
                    "Exerese_thoracique",
                    "Exerese_sein",
                    "Exerese_estomac",
                    "Exerese_foie",
                    "Exerese_oesophage",
                    "Exerese_pancreas",
                    "Exerese_rectum",
                    "Omentectomie_laparo",
                    "Omentectomie_coelio",
                    "Autres_resections_coelio",
                ]
            ]
            .notna()
            .any(axis=1)
        )

        codes_exerese = ccam.loc[idx_exerese, ["Code", "Libelle"]]
        codes_exerese.drop_duplicates(subset=["Code"], inplace=True)
        return codes_exerese

    def filter_by_visit(
        self, procedures: sparkDataFrame, path_filter_by_visit: str
    ) -> sparkDataFrame:
        # Get spark session
        spark, sql = get_spark_sql()
        visits_ids = (
            spark.read.parquet(path_filter_by_visit)  # documents
            .filter(F.col("note_class_source_value") == "RCP")  #
            .select("visit_occurrence_id")
            .drop_duplicates()
        )

        # Procedures of those visits
        procedures_visits = procedures.join(visits_ids, on="visit_occurrence_id")
        return procedures_visits
