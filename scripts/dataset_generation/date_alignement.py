import os
from typing import List, Optional

import pandas as pd
from confit import Cli

from oeciml.misc.utils import file_reader, get_name_config_and_project
from oeciml.pipelines.dataset_generation.dates.alignement import BaseDateTextAlignement

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="date_alignement")
def main(
    aligner: BaseDateTextAlignement,
    path_dates_to_match: List[str],
    col_name_dates_to_match: List[str],
    path_texts: Optional[str] = None,
    output_path: Optional[str] = None,
    path_dates_from_text: Optional[str] = None,
    cols=["person_id", "known_date", "label"],
    col_name_date="known_date",
    drop_duplicates=True,
    config_meta=None,
    **kwargs
):
    conf_name, project_name = get_name_config_and_project(config_meta)
    user = os.environ.get("USER")

    tmp_df = []
    for i, (path, col_name) in enumerate(
        zip(path_dates_to_match, col_name_dates_to_match), start=1
    ):
        df = file_reader(
            path,
            format_args=dict(user=user, project_name=project_name, conf_name=conf_name),
        )
        df["label"] = i
        df.rename(columns={col_name: col_name_date}, inplace=True)
        df = df[cols]
        tmp_df.append(df)

    dates_to_match = pd.concat(tmp_df)

    # Drop duplicates of same date and different label
    if drop_duplicates:
        dates_to_match.sort_values("label", ascending=False, inplace=True)
        dates_to_match.drop_duplicates(
            subset=["person_id", col_name_date], keep="first", inplace=True
        )

    texts = file_reader(
        path_texts,
        format_args=dict(user=user, project_name=project_name, conf_name=conf_name),
    )

    dates_from_text = file_reader(
        path_dates_from_text,
        format_args=dict(user=user, project_name=project_name, conf_name=conf_name),
    )

    dataset = aligner.generate_dataset(
        dates_to_match=dates_to_match,
        texts=texts,
        dates_from_text=dates_from_text,
        dates_to_match_col_name="known_date",
    )

    print("data generated")

    if output_path is not None:
        output_path = output_path.format(conf_name=conf_name)
        dataset.to_pickle(output_path)
        print("Results saved at:", output_path)
    else:
        return dataset


if __name__ == "__main__":
    app()
