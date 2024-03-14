import os
from typing import List

from confit import Cli

from oeciml.misc.utils import get_name_config_and_project
from oeciml.pipelines.documents import retrieve_docs_of_patient_set

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="retrieve_docs_of_patient_set")
def main(
    patient_set_or_path: str,
    db: str,
    doc_types: List[str] = ["RCP", "CR-ANAPATH"],
    cols_documents: List[str] = [
        "person_id",
        "visit_occurrence_id",
        "note_id",
        "note_datetime",
        "note_class_source_value",
        "note_text",
    ],
    output_path: str = None,
    config_meta=None,
):
    conf_name, project_name = get_name_config_and_project(config_meta)
    user = os.environ.get("USER")

    path_patients = patient_set_or_path.format(
        user=user, project_name=project_name, conf_name=conf_name
    )

    df = retrieve_docs_of_patient_set(
        patient_set_or_path=path_patients,
        db=db,
        doc_types=doc_types,
        cols_documents=cols_documents,
    )

    if output_path is not None:
        output_path = output_path.format(
            user=user, project_name=project_name, conf_name=conf_name
        )
        df.write.mode("overwrite").parquet(output_path)
        print("Results saved at:", output_path)


if __name__ == "__main__":
    app()
