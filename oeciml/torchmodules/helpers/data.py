from pathlib import Path
from typing import List, Optional

import datasets
import pandas as pd


class DFtoDataset:
    def __init__(
        self,
        data_path: Path,
        test_size: float = 0.1,
        seed: int = 42,
        text_col: str = "note_text",
        id_col: str = "text_id",
        span_cols: List[str] = ["offset_begin", "offset_end"],
        label_col: Optional[str] = "to_keep",
    ):
        self.data_path = data_path
        self.is_predict = label_col is None

        self.mapping = {
            "text": text_col,
            "id": id_col,
            "span_start": span_cols[0],
            "span_end": span_cols[1],
        }
        if not self.is_predict:
            self.mapping["label"] = label_col

        self.test_size = test_size
        self.seed = seed

    def prepare_data(self):

        df = pd.read_csv(self.data_path)

        data = df[list(self.mapping.values())]
        data.rename(
            columns={old: new for new, old in self.mapping.items()}, inplace=True
        )
        data.text.replace(
            {"ü": "u", "ö": "o"}, regex=True, inplace=True
        )  # Letters absent from camemBERT vocab
        dataset = data.to_dict(orient="list")

        self.dataset = datasets.Dataset.from_dict(dataset)

    def __call__(
        self,
    ):

        self.prepare_data()

        if self.is_predict:
            return self.dataset

        split_dataset = self.dataset.train_test_split(
            test_size=self.test_size, seed=self.seed
        )
        self.dataset = datasets.DatasetDict(
            {
                "train": split_dataset["train"],
                "val": split_dataset["test"],
            }
        )
        return self.dataset
