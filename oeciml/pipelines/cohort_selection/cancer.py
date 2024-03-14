from typing import Any, List, Optional, Union

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame
from pyspark.sql.window import Window

from oeciml.misc import retrieve_data
from oeciml.registry import registry

icd10_all_cancer = {
    "anus": ["C21", "D013"],
    "biliary_duct": ["C221", "C23", "C24", "D015", "D376"],
    "bladder": [
        "C66",
        "C67",
        "C68",
        "D090",
        "D091",
        "D412",
        "D413",
        "D414",
        "D417",
        "D419",
    ],
    "bowel": ["C17", "D014", "D372"],
    "breast": ["C50", "D05", "D486"],
    "cervix": ["C53", "D06"],
    "CNS": [
        "C70",
        "C71",
        "C720",
        "C722",
        "C723",
        "C728",
        "C729",
        "D42",
        "D430",
        "D431",
        "D432",
        "D434",
        "D437",
        "D439",
    ],
    "colon": [
        "C18",
        "C19",
        "D010",
        "D011",
        "D374",
        "D373",
        "C20",
        "D012",
        "D375",
    ],  # colon + rectum
    "CUP": ["C76", "C80", "C97", "D097", "D099", "D483", "D487", "D489"],
    "endometrium": ["C54", "C55", "D070", "D390"],
    "eye": ["C69", "D092"],
    "gastric": ["C16", "D002", "D371"],
    "head_neck": [
        "C0",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C30",
        "C31",
        "C32",
        "D000",
        "D020",
        "D370",
        "D380",
    ],
    "hodgkin_lymphoma": ["C81"],
    "kidney": ["C64", "C65", "D410", "D411"],
    "leukemia": [
        "C91",
        "C92",
        "C93",
        "C940",
        "C941",
        "C942",
        "C943",
        "C944",
        "C945",
        "C947",
        "C95",
    ],
    "liver": ["C220", "C222", "C223", "C224", "C227", "C229"],
    "lung": ["C33", "C34", "D021", "D022"],
    "mesothelioma": ["C45"],
    "myeloma": ["C90"],
    "nonhodgkin_lymphoma": ["C82", "C83", "C84", "C85", "C86"],
    "oesophagus": ["C15", "D001"],
    "osteosarcoma": ["C40", "C41", "D480"],
    "other_digestive": ["C26", "C48", "D017", "D019", "D377", "D379", "D484"],
    "other_endocrinial": [
        "C74",
        "C75",
        "D093",
        "D441",
        "D442",
        "D443",
        "D444",
        "D445",
        "D446",
        "D447",
        "D448",
        "D449",
    ],
    "other_gynecology": [
        "C51",
        "C52",
        "C57",
        "C58",
        "D071",
        "D072",
        "D073",
        "D392",
        "D397",
        "D399",
    ],
    "other_hematologic_malignancies": [
        "C46",
        "C88",
        "C96",
        "C946",
        "D45",
        "D46",
        "D47",
    ],
    "other_pneumology": [
        "C37",
        "C38",
        "C39",
        "D023",
        "D024",
        "D382" "D383",
        "D384",
        "D385",
        "D386",
    ],
    "other_skin": ["C44", "D04", "D485"],
    "other_urothelial": ["C60", "C63", "D074", "D076", "D407", "D409"],
    "ovary": ["C56", "D391"],
    "pancreas": [
        "C250",
        "C251",
        "C252",
        "C253",
        "C255",
        "C256",
        "C257",
        "C258",
        "C259",
    ],
    "PNS": ["C47", "C721", "C724", "C725", "D433", "D482"],
    "prostate": ["C61", "D075", "D400"],
    "soft_tissue": ["C49", "D481"],
    "skin": ["C43", "D03"],
    "testis": ["C62", "D401"],
    "thyroid": ["C73", "D440"],
}


@registry.cohort_selector("Cancer")
class CancerCohortSelector:
    def __init__(
        self,
        cancer_types: List[str],
        clean_period_years: int,
        db: str,
        start_date: str,
        end_date: str,
        double_cancer_exclusion: bool = True,
        db_type: str = "i2b2",
        claim_data_type: str = "AREM",
        diagnostic_types: Optional[Union[str, List[str]]] = None,
        cols_to_keep: List[str] = [
            "person_id",
            "cancer",
            "n_distinct_cancer",
        ],
        debug_sample: int = -1,
        **kwargs: Any,
    ) -> None:
        self.cancer_types = cancer_types
        self.clean_period_years = clean_period_years
        self.double_cancer_exclusion = double_cancer_exclusion
        self.claim_data_type = claim_data_type
        self.diagnostic_types = diagnostic_types
        self.cols_to_keep = cols_to_keep
        self.debug_sample = debug_sample
        self.db_type = db_type
        self.db = db
        self.start_date = start_date
        self.end_date = end_date

    def __call__(self, **kwargs: Any) -> sparkDataFrame:
        # All codes for all patients (with code of interest or not)
        icd10_patients0 = retrieve_data.retrieve_codes_icd(
            icd10_values=None,
            db=self.db,
            claim_data_type=self.claim_data_type,
            db_type=self.db_type,
            diagnostic_types=self.diagnostic_types,
        )

        # Make columns with the ICD10 code with 2 and 3 digits. Example: C40 & C401
        icd10_patients0 = icd10_patients0.withColumn(
            "condition_source_value_short_2",
            F.substring("condition_source_value", 1, 3),
        )
        icd10_patients0 = icd10_patients0.withColumn(
            "condition_source_value_short_3",
            F.substring("condition_source_value", 1, 4),
        )

        # Classify each ICD10 code into a family
        codes_cancer_pd = pd.DataFrame(
            icd10_all_cancer.items(), columns=["cancer", "code"]
        )
        codes_cancer_pd = codes_cancer_pd.explode("code", ignore_index=True)
        spark, sql = retrieve_data.get_spark_sql()
        codes_cancer = spark.createDataFrame(codes_cancer_pd)

        # We keep only lines related to cancer & add classification
        icd10_patients1 = icd10_patients0.join(
            codes_cancer.hint("broadcast"),
            on=(
                (icd10_patients0.condition_source_value_short_2 == codes_cancer.code)
                | (icd10_patients0.condition_source_value_short_3 == codes_cancer.code)
            ),
            how="inner",
        )

        # Count number of different families of ICD10 values for each patient
        # and detect patients only with one familly of codes (monocancer)
        windowSpec = Window.partitionBy("person_id")
        icd10_patients1 = icd10_patients1.withColumn(
            "n_distinct_cancer",
            F.size(F.collect_set("cancer").over(windowSpec)),
        )
        if self.double_cancer_exclusion:
            icd10_patients2 = icd10_patients1.where(F.col("n_distinct_cancer") == 1)
        else:
            icd10_patients2 = icd10_patients1

        # Time delta between two consecutive ICD10 codes for a patient & cancer
        windowSpec = Window.partitionBy(["person_id", "cancer"]).orderBy(
            "condition_start_datetime"
        )
        icd10_patients2 = icd10_patients2.withColumn(
            "condition_start_datetime_1",
            F.lag("condition_start_datetime").over(windowSpec),
        )
        icd10_patients2 = icd10_patients2.withColumn(
            "delta_code_date",
            F.datediff("condition_start_datetime", "condition_start_datetime_1"),
        )

        # Select only new patients
        clean_period_days = self.clean_period_years * 365
        windowSpec = Window.partitionBy(["person_id", "cancer"])
        icd10_patients2 = icd10_patients2.withColumn(
            "new_case",
            F.when(
                F.col("delta_code_date") <= clean_period_days,
                False,
            ).otherwise(True),
        )

        # Select only cancer of interest
        icd10_patients3 = icd10_patients2.filter(
            F.col("cancer").isin(self.cancer_types)
        )

        # Select only new cases
        icd10_patients4 = icd10_patients3.filter(F.col("new_case"))
        icd10_patients4 = icd10_patients4.drop("new_case")

        # coded between dates
        icd10_patients5 = icd10_patients4.filter(
            (F.col("condition_start_datetime") >= self.start_date)
            & (F.col("condition_start_datetime") < self.end_date)
        )

        # Select columns
        icd10_patients5 = icd10_patients5.select(self.cols_to_keep)

        # Drop duplicates
        icd10_patients6 = icd10_patients5.drop_duplicates(
            subset=["person_id", "cancer"]
        )

        # Sample patients
        if self.debug_sample > 0:
            icd10_patients6 = icd10_patients6.orderBy(F.rand()).limit(self.debug_sample)

        return icd10_patients6
