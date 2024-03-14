import os
from typing import Any, Dict

from confit import Cli

from oeciml.misc.utils import file_reader, get_name_config_and_project

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="split_set")
def main(stage: str, split_definitions: Dict[str, Any], config_meta=None, **kwargs):
    # Get variables
    output_path_train_pickle = split_definitions[stage]["output_path_train_pickle"]
    output_path_dev_or_test_pickle = split_definitions[stage][
        "output_path_dev_or_test_pickle"
    ]
    path_texts = split_definitions[stage]["path_texts"]
    splitter = split_definitions[stage]["splitter"]

    # Format args
    conf_name, project_name = get_name_config_and_project(config_meta)
    user = os.environ.get("USER")
    format_args = dict(user=user, project_name=project_name, conf_name=conf_name)

    # Import text to split
    texts = file_reader(path_texts, format_args=format_args)

    # Setup splitter
    splitter.setup(**format_args)

    # Call splitter
    other_texts, valid_or_test_texts = splitter(texts)

    # Export results
    if output_path_train_pickle is not None:
        output_path_train_pickle = output_path_train_pickle.format(conf_name=conf_name)
        other_texts.to_pickle(output_path_train_pickle)
        print("Train set exported :", output_path_train_pickle)

    if output_path_dev_or_test_pickle is not None:
        output_path_dev_or_test_pickle = output_path_dev_or_test_pickle.format(
            conf_name=conf_name
        )
        valid_or_test_texts.to_pickle(output_path_dev_or_test_pickle)
        print("Valid or test set exported :", output_path_train_pickle)

    print("All sets exported")


if __name__ == "__main__":
    app()
