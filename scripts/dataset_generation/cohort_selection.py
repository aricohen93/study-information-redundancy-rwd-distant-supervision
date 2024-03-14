import os
from typing import Any

from confit import Cli

from oeciml.misc.utils import get_name_config_and_project

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="cohort_selection")
def main(
    cohort_selector: Any,
    output_path: str = None,
    config_meta=None,
):
    df = cohort_selector()

    conf_name, project_name = get_name_config_and_project(config_meta)
    user = os.environ.get("USER")

    if output_path is not None:
        output_path = output_path.format(
            user=user, project_name=project_name, conf_name=conf_name
        )
        df.write.mode("overwrite").parquet(output_path)

        print("Results saved at:", output_path)


if __name__ == "__main__":
    app()
