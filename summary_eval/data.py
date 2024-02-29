import pandas as pd

from summary_eval.settings import SUMMARIES_TRAIN_PATH, PROMPTS_TRAIN_PATH
from summary_eval.util import logger


summary_df = pd.read_csv(SUMMARIES_TRAIN_PATH)
logger.info(f"Read {len(summary_df)} summaries from {SUMMARIES_TRAIN_PATH}")

prompts_df = pd.read_csv(PROMPTS_TRAIN_PATH)
logger.info(f"Read {len(prompts_df)} prompts from {PROMPTS_TRAIN_PATH}")
