from typing import Any, List

from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame
from pyspark.sql.window import Window


def flatten_list(nestedList) -> List[Any]:
    """"""
    # check if list is empty
    if not (bool(nestedList)):
        return nestedList

    # to check instance of list is empty or not
    if isinstance(nestedList[0], list):
        # call function with sublist as argument
        return flatten_list(*nestedList[:1]) + flatten_list(nestedList[1:])

    # call function with sublist as argument
    return nestedList[:1] + flatten_list(nestedList[1:])


def keep_one(df, col_date="note_datetime", how="first"):
    assert isinstance(df, sparkDataFrame)
    if how == "first":
        # Filter and keep only first
        windowSpec = Window.partitionBy(
            [
                "person_id",
            ]
        ).orderBy(F.col(col_date).asc())
    else:
        # Filter and keep only last
        windowSpec = Window.partitionBy(
            [
                "person_id",
            ]
        ).orderBy(F.col(col_date).desc())

    df = df.withColumn(
        "row",
        F.row_number().over(windowSpec),
    )

    condition = F.col("row") == 1
    df_filtered = df.filter(condition)
    df_filtered = df_filtered.drop("row")

    return df_filtered


def replace_by_new_text(span, new_text, span_start_char, span_end_char):
    new_span = span[:span_start_char] + new_text + span[span_end_char:]

    new_span_start_char = span_start_char
    new_span_end_char = new_span_start_char + len(new_text)

    return new_span, new_span_start_char, new_span_end_char


def wrap_replace_by_new_text(row, replacement_text):
    return replace_by_new_text(
        row.span, replacement_text, row.span_start_char, row.span_end_char
    )
