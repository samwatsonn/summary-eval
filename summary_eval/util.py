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


# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output log messages to the console
        # You can add more handlers like FileHandler to log to a file
    ]
)

# Create a logger instance
logger = logging.getLogger(__name__)



