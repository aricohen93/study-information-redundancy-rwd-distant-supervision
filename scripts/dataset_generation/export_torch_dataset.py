from confit import Cli

from oeciml.misc.utils import file_reader, get_name_config_and_project
from oeciml.torchmodules.data.dataset import DatasetCreator

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="export_torch_dataset")
def main(
    path_train_data: str,
    path_dev_data: str,
    dataset_creator: DatasetCreator,
    save_path: str,
    config_meta=None,
):
    conf_name, project_name = get_name_config_and_project(config_meta)
    path_train_data = path_train_data.format(conf_name=conf_name)
    path_dev_data = path_dev_data.format(conf_name=conf_name)

    train_data = file_reader(
        path_train_data,
        format_args=dict(conf_name=conf_name),
    )

    dev_data = file_reader(
        path_dev_data,
        format_args=dict(conf_name=conf_name),
    )

    dataset = dataset_creator.generate(train_data, dev_data)
    save_path = save_path.format(conf_name=conf_name)
    dataset.save_to_disk(save_path)

    print("Dataset saved at:", save_path)
    print(dataset)


if __name__ == "__main__":
    app()
