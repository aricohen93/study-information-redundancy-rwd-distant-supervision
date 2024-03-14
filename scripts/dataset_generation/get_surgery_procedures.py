import os
from typing import Optional

from confit import Cli

from oeciml.misc.utils import get_name_config_and_project
from oeciml.pipelines.structured.procedures import ProcedureSelector

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="procedures")
def main(
    selector: ProcedureSelector,
    output_path: str = None,
    path_filter_by_visit: Optional[str] = None,
    config_meta=None,
):
    conf_name, project_name = get_name_config_and_project(config_meta)
    user = os.environ.get("USER")

    if path_filter_by_visit is not None:
        path_filter_by_visit = path_filter_by_visit.format(
            user=user, project_name=project_name, conf_name=conf_name
        )

    df = selector(path_filter_by_visit=path_filter_by_visit)

    if output_path is not None:
        output_path = output_path.format(
            user=user, project_name=project_name, conf_name=conf_name
        )
        df.write.mode("overwrite").parquet(output_path)
        print("Results saved at:", output_path)


if __name__ == "__main__":
    app()
