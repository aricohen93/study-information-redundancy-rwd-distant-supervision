from typing import Any, Dict, Optional, Union

import pandas as pd

from oeciml.misc.retrieve_data import arrowConnector


def get_name_config_and_project(config_meta: Dict[str, Any]):
    name = config_meta["config_path"][0].stem

    project_name = config_meta["resolved_config"]["project"]["project_name"]
    return name, project_name


def is_hdfs_path(path: str):
    return path.startswith("hdfs")


def is_csv(path: str):
    return path.endswith("csv")


def format_path(path: str, format_args: Optional[Dict[str, str]] = None):
    if format_args is not None:
        return path.format(**format_args)
    else:
        return path


def file_reader(
    path: Optional[str], format_args: Optional[Dict[str, str]] = None, **kwds: Any
) -> Union[pd.DataFrame, None]:
    if path is not None:
        path = format_path(path, format_args)

        if is_csv(path):
            return pd.read_csv(path)
        elif is_hdfs_path(path):
            c = arrowConnector()
            return c.get_pd_table(path)
        else:
            return pd.read_pickle(path)
    else:
        return None
