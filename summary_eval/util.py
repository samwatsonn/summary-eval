import logging
import os
import pandas as pd

def split_x_y(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    return df.drop(columns=["content", "wording"]), df[["content", "wording"]]

def from_project_root(path: str) -> str:
    """Make sure that all paths are absolute."""
    project_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(project_dir)
    return os.path.join(root_dir, path)


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('summary-eval')
logger.setLevel(logging.DEBUG)  # Set the info level of the logger, not all loggers
