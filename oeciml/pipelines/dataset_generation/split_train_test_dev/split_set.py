from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame as pdDataFrame

from oeciml.misc.data_wrangling import flatten_list
from oeciml.misc.utils import file_reader
from oeciml.registry import registry


class Splitter:
    def __init__(
        self,
        note_class_source_values: Optional[List[str]] = None,
        paths_existing_data: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self.note_class_source_values = note_class_source_values
        self.paths_existing_data = paths_existing_data

    def setup(self, **kwargs) -> None:
        pass

    def filter_by_note_class(self, texts) -> pd.DataFrame:
        if self.note_class_source_values is not None:
            texts = texts.query(
                "note_class_source_value.isin(@self.note_class_source_values)"
            )
        return texts

    def filter_by_person_id(self, texts: pd.DataFrame) -> pd.DataFrame:
        if self.paths_existing_data is not None:
            avoid_person_id = []
            for path in self.paths_existing_data:
                df = file_reader(path)
                avoid_person_id.append(list(df.person_id.unique()))
                avoid_person_id = flatten_list(avoid_person_id)

            # Filter by person_id
            texts_filtered = texts.loc[~texts.person_id.isin(avoid_person_id)].copy()
            return texts_filtered
        else:
            return texts

    def __call__(self, texts: pd.DataFrame) -> Tuple[pdDataFrame, pdDataFrame]:
        raise NotImplementedError


@registry.splitter("SplitTest")
class SplitTest(Splitter):
    def __init__(
        self,
        n_test: int,
        seed: int,
        note_class_source_values: Optional[List[str]] = None,
        paths_existing_data: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(
            note_class_source_values=note_class_source_values,
            paths_existing_data=paths_existing_data,
        )

        self.n_test = n_test
        self.seed = seed

    def get_test_other_person_id(
        self,
        texts: pdDataFrame,
    ):
        # get unique person_id
        unique_person_id = texts.person_id.unique()

        # assign person_id to sets
        np.random.seed(self.seed)
        test_person_id = np.random.choice(unique_person_id, self.n_test, replace=False)
        other_person_id = np.setdiff1d(
            unique_person_id, test_person_id, assume_unique=True
        )

        return other_person_id, test_person_id

    def get_test_other_texts(
        self,
        texts,
        other_person_id,
        test_person_id,
    ):
        # filter dataset for the test and train set
        test = texts[texts.person_id.isin(test_person_id)]
        other = texts[texts.person_id.isin(other_person_id)]

        # Assign label 'test' and 'not test' in a new column
        test.insert(0, "set", "test", allow_duplicates=False)
        other.insert(0, "set", "not test", allow_duplicates=False)

        return other, test

    def __call__(
        self,
        texts,
    ):
        texts = self.filter_by_note_class(texts)
        texts = self.filter_by_person_id(texts)

        other_person_id, test_person_id = self.get_test_other_person_id(texts)

        return self.get_test_other_texts(texts, other_person_id, test_person_id)


@registry.splitter("SplitTrainExistingData")
class SplitTrainExistingData(Splitter):
    def __init__(
        self,
        paths_existing_data: Union[str, List[str]],
        note_class_source_values: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            note_class_source_values=note_class_source_values,
            paths_existing_data=paths_existing_data,
        )

    def __call__(self, texts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        texts = self.filter_by_note_class(texts)

        # Filter by person_id
        texts_train = self.filter_by_person_id(texts)

        return texts_train, None


@registry.splitter("SplitTrainTestDev")
class SplitTrainTestDev(Splitter):
    def __init__(
        self,
        n_test: int,
        n_dev: int,
        seed: int,
    ) -> None:
        super().__init__()

        self.n_dev = n_dev
        self.n_test = n_test
        self.seed = seed

    def get_train_test_dev_person_id(
        self,
        texts: pdDataFrame,
    ):
        # get unique person_id
        unique_person_id = texts.person_id.unique()

        # assign person_id to sets
        np.random.seed(self.seed)
        dev_test_person_id = np.random.choice(
            unique_person_id, self.n_dev + self.n_test, replace=False
        )
        dev_person_id = np.random.choice(dev_test_person_id, self.n_dev, replace=False)
        test_person_id = np.setdiff1d(
            dev_test_person_id, dev_person_id, assume_unique=True
        )

        train_person_id = np.setdiff1d(
            unique_person_id, dev_test_person_id, assume_unique=True
        )

        return train_person_id, test_person_id, dev_person_id

    def get_train_test_dev_texts(
        self, texts, train_person_id, test_person_id, dev_person_id
    ):
        # filter dataset for the test and train set
        test = texts[texts.person_id.isin(test_person_id)]
        train = texts[texts.person_id.isin(train_person_id)]

        # Assign label 'test' and 'train' in a new column
        test.insert(0, "set", "test", allow_duplicates=False)
        train.insert(0, "set", "train", allow_duplicates=False)

        # Create dev dataset
        dev = texts[texts.person_id.isin(dev_person_id)]

        # get last text for each person_id in the dev set
        dev = dev.sort_values("note_datetime").groupby("person_id").last().reset_index()

        # Assign label 'dev' in a new column
        dev.insert(0, "set", "dev", allow_duplicates=False)

        return train, test, dev

    def splitter_dev(
        self,
        texts,
    ):
        (
            train_person_id,
            test_person_id,
            dev_person_id,
        ) = self.get_train_test_dev_person_id(texts)
        return self.get_train_test_dev_texts(
            texts, train_person_id, test_person_id, dev_person_id
        )


@registry.splitter("SplitTrainTestDevAlignable")
class SplitTrainTestDevAlignable(Splitter):
    def __init__(
        self,
        seed,
        alignment_type,
        n_test: int = 200,
        n_dev_matched: int = 20,
        n_dev_without_cr: int = 20,  # cr for 'compte-rendu'
    ) -> None:
        super().__init__()

        self.seed = seed
        self.alignment_type = alignment_type
        self.n_test = n_test
        self.n_dev_matched = n_dev_matched
        self.n_dev_without_cr = n_dev_without_cr

    def get_train_test_dev_person_id(
        self,
        texts: pdDataFrame,
        dates_to_match: pdDataFrame,
        dates_to_match_col_name: str,
    ):
        # get unique person_id from texts
        unique_person_id = texts.person_id.unique()

        # set seed
        np.random.seed(self.seed)

        # test person_id
        test_person_id = np.random.choice(unique_person_id, self.n_test, replace=False)
        # other person_id
        train_dev_person_id = np.setdiff1d(
            unique_person_id, test_person_id, assume_unique=True
        )

        # align the whole train_dev set
        dataset = self.alignment_type.generate_dataset(
            dates_to_match,
            texts[texts["person_id"].isin(train_dev_person_id)],
            dates_to_match_col_name,
        )

        # for the dev set, we want n_dev_matched person_id with matched dates
        matched = dataset[dataset["label"] == True]  # noqa: E712
        train_dev_person_id_matched = matched.person_id.unique()
        dev_matched_person_id = np.random.choice(
            train_dev_person_id_matched, self.n_dev_matched, replace=False
        )

        # we keep the matched texts for each dev_matched_person_id
        dev_matched_note_id = matched[
            matched["person_id"].isin(dev_matched_person_id)
        ].note_id.to_list()

        # and n_dev_without_cr person_id without biopsy reports
        train_dev_person_id_without_cr = np.setdiff1d(
            train_dev_person_id, dates_to_match.person_id, assume_unique=True
        )
        dev_without_cr_person_id = np.random.choice(
            train_dev_person_id_without_cr, self.n_dev_without_cr, replace=False
        )

        # we now get the gathered dev set
        dev_person_id = np.concatenate(
            (dev_without_cr_person_id, dev_matched_person_id)
        )

        # train person_id : all person_ids that are not in the test or dev set
        train_person_id = np.setdiff1d(
            train_dev_person_id, dev_person_id, assume_unique=True
        )

        return (
            train_person_id,
            test_person_id,
            dev_matched_person_id,
            dev_without_cr_person_id,
            dev_matched_note_id,
        )

    def get_train_test_dev_texts(
        self,
        texts: pdDataFrame,
        train_person_id,
        test_person_id,
        dev_matched_person_id,
        dev_without_cr_person_id,
        dev_matched_note_id,
    ):
        # filter dataset for the test and train set
        test = texts[texts.person_id.isin(test_person_id)]
        train = texts[texts.person_id.isin(train_person_id)]

        # Assign label 'test' and 'train' in a new column
        test.insert(0, "set", "test", allow_duplicates=False)
        train.insert(0, "set", "train", allow_duplicates=False)

        # Create dev dataset
        dev_without_cr = texts[texts.person_id.isin(dev_without_cr_person_id)]
        dev_matched = texts[texts.note_id.isin(dev_matched_note_id)]
        dev = pd.concat([dev_matched, dev_without_cr], ignore_index=True)

        # get last text for each person_id in the dev set
        dev = dev.sort_values("note_datetime").groupby("person_id").last().reset_index()

        # Assign label 'dev' in a new column
        dev.insert(0, "set", "dev", allow_duplicates=False)
        # and a specific column to know if it is a matched or a without cr text
        dev.insert(
            1,
            "alignment",
            [
                "dev matched" if x in (dev_matched_person_id) else "dev without cr"
                for x in dev["person_id"]
            ],
            allow_duplicates=False,
        )

        return train, test, dev

    def splitter_dev(
        self,
        texts: pdDataFrame,
        dates_to_match: pdDataFrame,
        dates_to_match_col_name: str,
    ):
        (
            train_person_id,
            test_person_id,
            dev_matched_person_id,
            dev_without_cr_person_id,
            dev_matched_note_id,
        ) = self.get_train_test_dev_person_id(
            texts, dates_to_match, dates_to_match_col_name
        )
        return self.get_train_test_dev_texts(
            texts,
            train_person_id,
            test_person_id,
            dev_matched_person_id,
            dev_without_cr_person_id,
            dev_matched_note_id,
        )


@registry.splitter("SplitTrainDevAlignable")
class SplitTrainDevAlignable(Splitter):
    def __init__(
        self,
        path_dates_to_match: str,
        path_aligned_texts: str,
        n_dev_matched: int,
        n_dev_without_cr: int,
        seed: int = 29,
    ) -> None:
        super().__init__()

        self.path_dates_to_match = path_dates_to_match
        self.path_aligned_texts = path_aligned_texts
        self.seed = seed
        self.n_dev_matched = n_dev_matched
        self.n_dev_without_cr = n_dev_without_cr

        self.user = None
        self.project_name = None
        self.conf_name = None

    def setup(
        self,
        user: Optional[str] = None,
        project_name: Optional[str] = None,
        conf_name: Optional[str] = None,
        **kwargs
    ):
        self.user = user
        self.project_name = project_name
        self.conf_name = conf_name

    def __call__(self, texts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        texts_aligned = file_reader(
            self.path_aligned_texts, format_args=dict(conf_name=self.conf_name)
        )

        dates_to_match = file_reader(
            self.path_dates_to_match,
            format_args=dict(
                user=self.user, project_name=self.project_name, conf_name=self.conf_name
            ),
        )

        print("Texts imported")

        # Set seed
        np.random.seed(self.seed)

        # type standardization
        texts_aligned.person_id = texts_aligned.person_id.astype(str)
        texts.person_id = texts.person_id.astype(str)

        texts_aligned.note_id = texts_aligned.note_id.astype(str)
        texts.note_id = texts.note_id.astype(str)

        # for the dev set, we want n_dev_matched person_id with matched dates
        matched = texts_aligned[texts_aligned["label"] == True]  # noqa: E712
        dev_matched_person_id = np.random.choice(
            matched.person_id.unique(), self.n_dev_matched, replace=False
        )
        dev_matched_note_id = matched[
            matched.person_id.isin(dev_matched_person_id)
        ].note_id

        # all others person_id are kept for the train set
        train_person_id = np.setdiff1d(
            texts_aligned.person_id.unique(), dev_matched_person_id, assume_unique=True
        )

        # we also want n_dev_without_cr person_id without biopsy reports
        person_id_without_cr = np.setdiff1d(
            texts.person_id.unique(),
            dates_to_match.person_id.unique(),
            assume_unique=True,
        )
        dev_without_cr_person_id = np.random.choice(
            person_id_without_cr, self.n_dev_without_cr, replace=False
        )
        # # we now get the gathered dev set
        # dev_person_id = np.concatenate(
        #     (dev_without_cr_person_id, dev_matched_person_id)
        # )  # FIXME

        # let's now get the texts for each set based on the selected person_ids
        train_aligned = texts_aligned[texts_aligned.person_id.isin(train_person_id)]
        train_aligned["set"] = "train"
        train_aligned.reset_index(inplace=True, drop=True)
        train_aligned["text_id"] = [i for i in range(len(train_aligned))]

        train = texts[texts.person_id.isin(train_person_id)]
        train["set"] = "train"
        train.reset_index(inplace=True, drop=True)

        dev_without_cr = texts[texts.person_id.isin(dev_without_cr_person_id)]
        dev_matched = texts[texts.note_id.isin(dev_matched_note_id)]
        dev = pd.concat([dev_matched, dev_without_cr])

        # get last text for each person_id in the dev set
        dev = dev.sort_values("note_datetime").groupby("person_id").last().reset_index()

        dev["set"] = "dev"
        dev.insert(
            1,
            "alignment",
            [
                "dev matched" if x in (dev_matched_person_id) else "dev without cr"
                for x in dev["person_id"]
            ],
            allow_duplicates=False,
        )
        dev.reset_index(inplace=True, drop=True)
        print("Sets created")

        return train_aligned, dev
