import pandas as pd


def any_nans(data: list[pd.DataFrame]|pd.DataFrame) -> bool:
    """
    Calculate if a list of data frames contains any nans.

    :param data: The list of data frame to process
    :type data: list[pd.DataFrame]
    :return: Bool if any data frame of the list contains any nan values
    :rtype: bool
    """
    return any((df.isna().any(axis=None) > 0) for df in data)

def construct_updated_data_frame(
    data: list[pd.DataFrame]|pd.DataFrame,
    data_one_model: list[pd.DataFrame]|pd.DataFrame,
    model_label: str,
) -> list[pd.DataFrame]:
    """
    Construct an updated data frame providing the previous data frame and the data frame of the current model call.

    Also adds the model label to the data frame.
    """
    if isinstance(data_one_model, list) and data_one_model == []:  # nothing to update
        return data

    if isinstance(data_one_model, pd.DataFrame):
        data_one_model = [data_one_model]

    if isinstance(data, pd.DataFrame):
        data = [data]

    # extend the data we have read so far to match the new ensemble numbers
    if len(data) == 1 and len(data_one_model) > 1:
        data = data * len(data_one_model)
    elif len(data) != len(data_one_model):
        msg = f"Tried to combine models with different ensemble numbers: {len(data)} and {len(data_one_model)}!"
        raise ValueError(msg)

    for i, _ in enumerate(data_one_model):
        data_one_model[i]["model"] = model_label
        data_one_model[i].loc[data_one_model[i].isna().any(axis=1), "model"] = None
        data[i] = data[i].combine_first(data_one_model[i])

    return data
