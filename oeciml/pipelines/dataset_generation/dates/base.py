from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import spacy
from edsnlp.processing import pipe
from pandas import DataFrame as pdDataFrame
from pendulum.datetime import DateTime
from pyspark.sql import types as T
from pyspark.sql.dataframe import DataFrame as sparkDataFrame
from pyspark.sql.types import TimestampType
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from oeciml.spacy.utils.align import get_span_text_and_offsets


@Language.component("exclude_semi_structured_dates")
def exclude_semi_structured_dates(doc: Doc) -> Doc:
    """
    Spots semi-structured dates and excludes them.

    Parameters
    ----------
    doc : Doc
        spaCy Doc object

    Returns
    -------
    doc : Doc
        spaCy Doc object, with semi-structured dates excluded
    """

    spans = doc.spans["dates"]

    if len(spans) > 0:
        for date in spans:
            if date.start == 0:
                if doc[date.end].like_num:
                    for token in date:
                        token._.excluded = True
                elif (doc[date.end].is_space) and (doc[date.end + 1].like_num):
                    for token in date:
                        token._.excluded = True

            elif date.end == (len(doc) - 1):
                if doc[date.start - 1].like_num:
                    for token in date:
                        token._.excluded = True
                elif (doc[date.start - 1].is_space) and (doc[date.start - 2].like_num):
                    for token in date:
                        token._.excluded = True

            else:
                if doc[date.start - 1].like_num or doc[date.end].like_num:
                    for token in date:
                        token._.excluded = True
                elif (
                    (doc[date.start - 1].is_space) and (doc[date.start - 2].like_num)
                ) or ((doc[date.end].is_space) and (doc[date.end + 1].like_num)):
                    for token in date:
                        token._.excluded = True

    return doc


class getDatesfromText:
    def __init__(
        self,
        n_before: int = 30,
        n_after: int = 45,
        ignore_excluded: bool = True,
        drop_na: bool = True,
        attr: str = "NORM",
        tz: str = None,
    ) -> None:
        self.n_before = n_before
        self.n_after = n_after
        self.ignore_excluded = ignore_excluded
        self.drop_na = drop_na
        self.attr = attr
        self.tz = tz

    def __call__(
        self,
        texts: Union[pdDataFrame, sparkDataFrame],
    ) -> pdDataFrame:
        # Initialize spacy nlp object
        nlp = self.get_nlp()

        # Get dictionary of result data types
        dtypes = self.get_dtypes(texts)

        # Extract dates
        entities = pipe(
            texts,
            nlp,
            n_jobs=-2,  #
            results_extractor=self.get_dates,
            context=["note_datetime"],
            dtypes=dtypes,
        )

        # Drop non parsed entities
        if self.drop_na:
            entities = entities.dropna(subset=["date_dt"])

        # Add person_id
        entities = self.add_person_id(entities, texts)

        return entities

    @staticmethod
    def get_nlp(
        config_norm: Dict[str, bool] = dict(
            lowercase=False,
            accents=False,
            quotes=False,
            pollution=True,
        )
    ):
        nlp = spacy.blank("eds")
        nlp.add_pipe("eds.normalizer", config=config_norm)
        nlp.add_pipe("eds.sentences")
        nlp.add_pipe("eds.dates")
        nlp.add_pipe("exclude_semi_structured_dates")
        return nlp

    def get_dtypes(
        self, df: Union[pdDataFrame, sparkDataFrame]
    ) -> Optional[Dict[str, Any]]:
        # Define data types
        if isinstance(df, sparkDataFrame):
            str_type = T._infer_type("")
            int_type = T._infer_type(1)
            dtypes = dict(
                note_id=str_type,
                lexical_variant=str_type,
                date_dt=TimestampType(),
                text_start_char=int_type,
                text_end_char=int_type,
                span=str_type,
                span_start_char=int_type,
                span_end_char=int_type,
            )
        else:
            dtypes = None
        return dtypes

    def get_dates(
        self,
        doc: Doc,
    ) -> List[Dict[str, Any]]:
        """
        Function used well Paralellizing tasks via joblib
        This functions will return a list of dictionaries of all extracted entities
        """

        dates = []
        if "dates" in list(doc.spans.keys()):
            dates = []
            for date in doc.spans["dates"]:
                dates.append(self.pick_date(date))

        return dates

    def pick_date(
        self,
        date: Span,
    ) -> Dict[str, Any]:
        """Function to return different attributes of a date as a dictionary
        Useful to export to a pandas dataframe

        Parameters
        ----------
        date : span of date

        Returns
        -------
        Dictionary with different attributes of the entity
        """

        if date.doc.has_extension("note_datetime"):
            visit_start_date = date.doc._.get("note_datetime")
        else:
            visit_start_date = None

        date_dt = self._to_datetime_getter(date, visit_start_date)

        snippet_text, (span_start_char, span_end_char) = get_span_text_and_offsets(
            date,
            n_before=self.n_before,
            n_after=self.n_after,
            return_type="text",
            mode="token",
            ignore_excluded=self.ignore_excluded,
            attr=self.attr,
        )

        if self.ignore_excluded:
            if date[0]._.excluded:
                date_dt = None

        return dict(
            note_id=date.doc._.note_id,
            lexical_variant=date.text,
            date_dt=date_dt,
            text_start_char=date.start_char,
            text_end_char=date.end_char,
            span=snippet_text,
            span_start_char=span_start_char,
            span_end_char=span_end_char,
        )

    @staticmethod
    def add_person_id(
        entities: Union[pdDataFrame, sparkDataFrame],
        texts: Union[pdDataFrame, sparkDataFrame],
    ):
        if isinstance(entities, sparkDataFrame):
            entities = entities.join(
                texts.select(["note_id", "person_id"]),
                how="left",
                on="note_id",
            )
            entities = entities.toPandas()
        else:
            entities = entities.merge(
                texts[["note_id", "person_id"]],
                how="left",
                validate="many_to_one",
                on="note_id",
            )
        return entities

    def _to_datetime_getter(
        self,
        date: Span,
        visit_start_date: Union[datetime, DateTime],
    ):
        try:
            dt = date._.date.to_datetime(
                note_datetime=visit_start_date, infer_from_context=True, tz=self.tz
            )

            if isinstance(dt, (datetime, DateTime)):
                return dt
            else:
                return None

        except:  # noqa: E722
            return None
