import os
from typing import List, Optional

from confit import Cli

from oeciml.misc.utils import get_name_config_and_project
from oeciml.pipelines.documents import get_biopsy_reports

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="get_biopsy_reports")
def main(
    documents_or_path: str,
    db: str,
    ghm_codes_of_stay: Optional[List[str]] = ["M", "Z", "K"],
    adicap_sampling_modes: Optional[List[str]] = [
        "BIOPSIE CHIRURGICALE",
        "CYTOPONCTION NON GUIDEE PAR IMAGERIE",
        "CYTOPONCTION GUIDEE PAR IMAGERIE",
        "HISTOPONCTION GUIDEE PAR IMAGERIE",
        "PONCTION BIOPSIE ET BIOPSIE INSTRUMENTALE NON GUIDEE PAR IMAGERIE",
        "BIOPSIE TRANSVASCULAIRE",
    ],
    return_cols: List[str] = [
        "note_id",
        "note_text",
        "note_datetime",
        "sampling_date",
        "reception_date",
        "note_class_source_value",
        "person_id",
        "visit_occurrence_id",
    ],
    output_path: str = None,
    config_meta=None,
):
    conf_name, project_name = get_name_config_and_project(config_meta)
    user = os.environ.get("USER")

    path_documents = documents_or_path.format(
        user=user, project_name=project_name, conf_name=conf_name
    )

    df = get_biopsy_reports(
        documents_or_path=path_documents,
        db=db,
        ghm_codes_of_stay=ghm_codes_of_stay,
        adicap_sampling_modes=adicap_sampling_modes,
        return_cols=return_cols,
    )

    if output_path is not None:
        output_path = output_path.format(
            user=user, project_name=project_name, conf_name=conf_name
        )
        df.write.mode("overwrite").parquet(output_path)
        print("Results saved at:", output_path)


if __name__ == "__main__":
    app()
