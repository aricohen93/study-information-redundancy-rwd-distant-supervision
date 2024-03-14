import os
from pathlib import Path

import pandas as pd
from labeltool.labelling import Labelling

proxy = "http://proxym-inter.aphp.fr:8080"
os.environ["http_proxy"] = proxy
os.environ["HTTP_PROXY"] = proxy
os.environ["https_proxy"] = proxy
os.environ["HTTPS_PROXY"] = proxy


class EmmanuelleLabelling:
    def __init__(
        self,
        window_snippet=25,
        labels_height=150,
    ) -> None:
        super().__init__()

        self.window_snippet = window_snippet
        self.labels_height = labels_height

    def prepare_data_to_label(self, data_pre_labeled_path, documents_path, folder_path):
        # load pre labeled data
        data = pd.read_pickle(data_pre_labeled_path)
        data.dropna()

        entities_path = Path(folder_path) / "entities.pickle"
        texts_path = Path(folder_path) / "texts.pickle"

        if entities_path.exists() and texts_path.exists():
            print("Using existing files")
            return

        # create entities
        entities = data.rename(
            columns={"text_start_char": "offset_begin", "text_end_char": "offset_end"}
        )
        entities["label_name"] = "Biopsie/Cytoponction"
        entities["label_value"] = False
        entities["diagnostic"] = "pas de diagnostic"
        entities["type"] = "biopsie"
        entities.label_value.fillna(False, inplace=True)

        # save entities
        entities.to_pickle(entities_path)

        print("entities saved")

        # load documents
        docs = pd.read_pickle(documents_path)

        # create texts
        texts = docs[["note_id", "note_text"]]

        # make sure all docs are in common
        # texts['note_id']= texts['note_id'].astype('int')
        texts = texts.loc[texts.note_id.isin(entities.note_id.unique())]
        # assign note_ids as titles
        texts["title"] = texts.note_id.astype(str)
        # save texts
        texts.to_pickle(texts_path)

        print("texts saved")

    def define_parameters_labeltool(self):
        labels_params = [
            {
                "label_name": "Biopsie/Cytoponction",
                "color": "from_value",
                "selection_type": "binary",
                "default_value": True,
            }
        ]

        modifiers_params = [
            {
                "modifier_name": "type",
                "selection_type": "list",
                "selection_values": ["biopsie", "cytoponction"],
            },
            {
                "modifier_name": "diagnostic",
                "selection_type": "list",
                "selection_values": [
                    "initial",
                    "rechute",
                    "autre",
                    "pas de diagnostic",
                ],
            },
        ]

        global_params = [
            {
                "global_label_name": "Remarque",
                "selection_type": "text",
                "from": False,
                "value": "",
            }
        ]

        return labels_params, modifiers_params, global_params

    def instantiate_labelling(
        self,
        data_pre_labeled_path,
        documents_path,
        folder_path,
        window_snippet,
        labels_height,
    ):
        # create files and save them in folder
        self.prepare_data_to_label(data_pre_labeled_path, documents_path, folder_path)

        # define parameters
        (
            labels_params,
            modifiers_params,
            global_params,
        ) = self.define_parameters_labeltool()

        print("parameters defined")

        # instantiate Labelling
        LabellingTool = Labelling(
            folder_path=folder_path,
            labels_params=labels_params,
            modifiers_params=modifiers_params,
            global_labels_params=global_params,
            window_snippet=self.window_snippet,
            labels_height=self.labels_height,
        )

        print("Labelling instantiated")
        print("You can now annotate! :)")

        return LabellingTool
