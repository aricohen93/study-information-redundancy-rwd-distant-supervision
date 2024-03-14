from pathlib import Path
from typing import Optional, Union

import pandas as pd
from pandas import DataFrame as pdDataFrame
from pyspark.sql.dataframe import DataFrame as sparkDataFrame
from scipy.stats import norm

from oeciml.misc.data_wrangling import wrap_replace_by_new_text
from oeciml.pipelines.dataset_generation.dates.base import getDatesfromText
from oeciml.registry import registry


class BaseDateTextAlignement:
    def __init__(
        self,
        n_before: int = 30,
        n_after: int = 45,
        ignore_excluded: bool = False,
        drop_na: bool = True,
        attr: str = "NORM",
        tz: str = None,
        path_dates_from_text: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.n_before = n_before
        self.n_after = n_after
        self.ignore_excluded = ignore_excluded
        self.drop_na = drop_na
        self.attr = attr
        self.tz = tz
        self.path_dates_from_text = path_dates_from_text

    def generate_dataset(
        self,
        dates_to_match: Union[pdDataFrame, str, Path],
        texts: Optional[Union[pdDataFrame, sparkDataFrame, str, Path]],
        dates_to_match_col_name: str,
        dates_from_text: Optional[pd.DataFrame] = None,
    ) -> pdDataFrame:
        self.dates_to_match = dates_to_match
        self.texts = texts
        self.dates_to_match_col_name = dates_to_match_col_name

        if texts is not None:
            dates_from_text = getDatesfromText(
                n_before=self.n_before,
                n_after=self.n_after,
                ignore_excluded=self.ignore_excluded,
                drop_na=self.drop_na,
                attr=self.attr,
                # tz=self.tz,
            )(self.texts)
            if self.path_dates_from_text is not None:
                dates_from_text.to_pickle(self.path_dates_from_text)
        else:
            assert dates_from_text is not None

        dates_from_text = self.count_date_occurrence_by_doc(dates_from_text)

        dataset = self.alignment(
            dates_to_match=self.dates_to_match,
            dates_from_text=dates_from_text,
        )
        dataset.drop_duplicates(subset=["span", "person_id", "label"], inplace=True)

        # add column 'set' (train, test, dev) if there was one in texts
        if isinstance(texts, pd.DataFrame):
            if "set" in texts.columns:
                dataset = dataset.merge(
                    texts[["set", "note_id"]], how="left", on="note_id"
                )
                first_column = dataset.pop("set")
                dataset.insert(0, "set", first_column)

        return dataset

    def count_date_occurrence_by_doc(self, dates_from_text: pdDataFrame) -> pdDataFrame:
        # Count Dates by doc
        count_dates_by_document = dates_from_text.groupby(
            ["note_id", "date_dt"], as_index=False
        ).size()
        count_dates_by_document.rename(columns={"size": "count_date"}, inplace=True)
        dates_from_text = dates_from_text.merge(
            count_dates_by_document,
            on=["note_id", "date_dt"],
            how="left",
            validate="many_to_one",
        )
        return dates_from_text

    def localize_tz(self, df, col):
        if df[col].dt.tz is None:
            df[col] = df[col].dt.tz_localize(self.tz)

    def match_dates(
        self,
        dates_to_match: pdDataFrame,
        dates_from_text: pdDataFrame,
        direction: str = "backward",
        tolerance: pd.Timedelta = pd.to_timedelta("0D"),
        threshold: str = "1930-01-01",
    ) -> pdDataFrame:
        # Sort dates
        dates_to_match.sort_values(self.dates_to_match_col_name, inplace=True)
        dates_from_text.sort_values("date_dt", inplace=True)

        # Cast to datetime
        dates_to_match[self.dates_to_match_col_name] = pd.to_datetime(
            dates_to_match[self.dates_to_match_col_name], errors="coerce"
        )
        # Drop NaT rows
        dates_to_match.dropna(subset=[self.dates_to_match_col_name], inplace=True)

        # Localize TimeZone
        if self.tz is not None:
            self.localize_tz(dates_to_match, self.dates_to_match_col_name)
            self.localize_tz(dates_from_text, "date_dt")

        cols = [
            "person_id",
            self.dates_to_match_col_name,
        ]
        if "label" in dates_to_match.columns:
            cols.append("label")

        # Merge both dataframes
        alignement = pd.merge_asof(
            dates_from_text,
            dates_to_match[cols],
            by="person_id",
            left_on="date_dt",
            right_on=self.dates_to_match_col_name,
            direction=direction,
            tolerance=tolerance,
            suffixes=("", "_dates_to_match"),
        )

        if "label" not in dates_to_match.columns:
            # Label True dates with match
            alignement["label"] = alignement[self.dates_to_match_col_name].notna()
        else:
            alignement["label"] = alignement["label"].fillna(0)

        # Number of matches per doc
        alignement_doc_info = alignement.groupby("note_id").agg(
            matched_date_in_doc=("label", "sum")
        )

        alignement = alignement.merge(alignement_doc_info, on="note_id")
        # Filter
        alignement = alignement.loc[alignement.date_dt > threshold]

        # Keep only docs with matched dates
        alignement = alignement.query("matched_date_in_doc>0").copy()

        # Reset index
        alignement.reset_index(inplace=True, drop=True)

        return alignement

    @staticmethod
    def mask_entity(df):
        df[
            ["masked_span", "masked_span_start_char", "masked_span_end_char"]
        ] = df.apply(
            wrap_replace_by_new_text,
            axis=1,
            **{"replacement_text": "<mask>"},
            result_type="expand",
        )

        return df

    def alignment(
        self,
        dates_to_match: pdDataFrame,
        dates_from_text: pdDataFrame,
        **kwargs,
    ) -> pdDataFrame:
        raise NotImplementedError


@registry.aligner("DateTextAlignementProximity")
class DateTextAlignementProximity(BaseDateTextAlignement):
    def __init__(
        self,
        n_before: int = 30,
        n_after: int = 45,
        ignore_excluded: bool = False,
        drop_na: bool = True,
        attr: str = "NORM",
        tz: str = None,
        path_dates_from_text: Optional[str] = None,
        mask_dates: bool = True,
        random_state: int = 0,
        scale: int = 1500,
        false_examples_multiplier: int = 1,
    ) -> None:
        super().__init__(
            n_before, n_after, ignore_excluded, drop_na, attr, tz, path_dates_from_text
        )
        self.scale = scale
        self.random_state = random_state
        self.mask_dates = mask_dates
        self.false_examples_multiplier = false_examples_multiplier

    def get_positive_examples(self, dates_to_match, dates_from_text):
        unique_dates = dates_from_text.query("count_date==1").copy()

        # Merge dates extractions
        alignement = self.match_dates(
            dates_from_text=unique_dates,
            dates_to_match=dates_to_match,
        )

        return alignement

    def get_negative_examples(self, alignement, n):
        # Sample False examples
        alignement["position"] = alignement[["text_start_char", "text_end_char"]].mean(
            axis=1
        )

        # apply weight
        alignement = alignement.groupby("note_id").apply(self.weight)
        alignement = alignement.dropna(subset=["label"], inplace=False)
        alignement.label = alignement.label.astype(int)
        # print(dates_from_text.label)
        tmp = alignement.query("label == 0").copy()
        tmp.drop_duplicates(subset=["span", "person_id", "label"], inplace=True)

        not_matches = tmp.sample(n=n, random_state=self.random_state, weights=tmp.p)

        # not_matches.drop(columns=["position", "p"], inplace=True)
        return not_matches

    def weight(self, df):
        locs = df.query("label!=0").position.values
        df["p"] = df.position.apply(
            lambda x: norm.pdf(x, loc=locs, scale=self.scale).sum()
        )
        return df

    def alignment(
        self,
        dates_to_match: pdDataFrame,
        dates_from_text: pdDataFrame,
    ) -> pdDataFrame:
        # Get Positive class Examples
        alignement = self.get_positive_examples(
            dates_to_match=dates_to_match,
            dates_from_text=dates_from_text,
        )

        # Matches
        matches = alignement.query("label!=0").copy()
        matches.drop_duplicates(subset=["span", "person_id", "label"], inplace=True)

        # Sample non matching dates
        n1 = len(matches.query("label==1"))
        # n0 = len(matches.query("label==0"))

        not_matches = self.get_negative_examples(
            alignement=alignement,
            # dates_from_text=dates_from_text,
            n=(n1) * self.false_examples_multiplier,
        )

        # Join both
        examples = pd.concat([matches, not_matches])
        examples.reset_index(inplace=True, drop=True)
        examples.reset_index(inplace=True, drop=False)
        examples.rename(columns={"index": "text_id"}, inplace=True)

        # Drop cols
        examples.drop(
            columns=["date_dt", "count_date", self.dates_to_match_col_name],
            inplace=True,
        )

        if self.mask_dates:
            examples = self.mask_entity(examples)

        examples.label = examples.label.astype(int)

        return examples


@registry.aligner("DateTextAlignementV1")
class DateTextAlignementV1(DateTextAlignementProximity):
    def __init__(
        self,
        n_before: int = 30,
        n_after: int = 45,
        ignore_excluded: bool = False,
        drop_na: bool = True,
        attr: str = "NORM",
        tz: str = None,
        path_dates_from_text: Optional[str] = None,
        mask_dates: bool = True,
        random_state: int = 0,
        false_examples_multiplier: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            n_before,
            n_after,
            ignore_excluded,
            drop_na,
            attr,
            tz,
            path_dates_from_text,
            mask_dates,
            random_state,
            false_examples_multiplier,
        )

    def get_negative_examples(self, alignement, n):
        alignement = alignement.dropna(subset=["label"], inplace=False)
        alignement.label = alignement.label.astype(int)
        tmp = alignement.query("label == 0").copy()
        tmp.drop_duplicates(subset=["span", "person_id", "label"], inplace=True)

        not_matches = tmp.sample(n=n, random_state=self.random_state)

        return not_matches
