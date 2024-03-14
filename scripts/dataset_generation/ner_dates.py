import os

from confit import Cli

from oeciml.misc.utils import file_reader, get_name_config_and_project
from oeciml.pipelines.dataset_generation.dates.base import getDatesfromText

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="ner_dates")
def main(
    path_texts: str,
    output_path: str = None,
    n_before: int = 30,
    n_after: int = 45,
    ignore_excluded: bool = True,
    drop_na: bool = True,
    attr: str = "NORM",
    config_meta=None,
):
    conf_name, project_name = get_name_config_and_project(config_meta)
    user = os.environ.get("USER")

    texts = file_reader(
        path_texts,
        format_args=dict(user=user, project_name=project_name, conf_name=conf_name),
    )

    dates_from_text = getDatesfromText(
        n_before=n_before,
        n_after=n_after,
        ignore_excluded=ignore_excluded,
        drop_na=drop_na,
        attr=attr,
    )(texts)

    if output_path is not None:
        output_path = output_path.format(
            user=user, project_name=project_name, conf_name=conf_name
        )
        dates_from_text.to_pickle(output_path)

        print("Results saved at:", output_path)


if __name__ == "__main__":
    app()
