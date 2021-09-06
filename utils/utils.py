import pandas as pd
from pandas.core.common import flatten
from collections import OrderedDict


# Function to fill missing seconds (for each stock id)
def fill_missing_seconds(df):
    time_ids = list(OrderedDict.fromkeys(df["time_id"]))

    filled_df = pd.DataFrame(index=range(len(time_ids) * 600), columns=df.columns)

    all_seconds_in_bucket = list(flatten([range(600) for i in range(len(time_ids))]))
    filled_df["seconds_in_bucket"] = all_seconds_in_bucket

    all_time_ids = list(flatten([[time_ids[i]] * 600 for i in range(len(time_ids))]))
    filled_df["time_id"] = all_time_ids

    filled_df = pd.merge(filled_df, df, on=["time_id", "seconds_in_bucket"], how="left", suffixes=("_to_move", ""))

    to_remove_columns = [column for column in filled_df.columns if "to_move" in column]
    filled_df = filled_df.drop(to_remove_columns, axis=1)

    filled_df = filled_df.fillna(method="ffill")

    return filled_df
