from typing import Dict

import numpy as np
import pandas as pd


def load_data(
    n_steps_ahead: int = 10, print_result: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    data = np.load("dataset.npy")
    n_instances = 10_000
    n_steps = 50
    split = [7_000, 9_000]

    dataset = {}
    df = pd.DataFrame.from_dict(
        {
            "train": {
                "X": data[: split[0], :-1],
                "Y": data[: split[0], -1],
            },
            "valid": {
                "X": data[split[0] : split[1], :-1],
                "Y": data[split[0] : split[1], -1],
            },
            "test": {
                "X": data[split[1] :, -1],
                "Y": data[split[1] :, -1],
            },
        },
        orient="index",
    )
    dataset["single_step"] = df

    df["X"]["train"] = data[: split[0], :-n_steps_ahead]
    df["Y"]["train"] = data[: split[0], -n_steps_ahead:, 0]
    df["X"]["valid"] = data[split[0] : split[1], :-n_steps_ahead]
    df["Y"]["valid"] = data[split[0] : split[1], -n_steps_ahead:, 0]
    df["X"]["test"] = data[split[1] :, :-n_steps_ahead]
    df["Y"]["test"] = data[split[1] :, -n_steps_ahead:, 0]
    dataset["multi_horizon"] = df

    y_sts = np.empty((n_instances, n_steps - n_steps_ahead + 1, n_steps_ahead))
    for i_channel in range(n_steps_ahead):
        y_sts[:, :, i_channel] = data[
            :, (i_channel + 1) : (i_channel + 2 + n_steps - n_steps_ahead), 0
        ]
    y_train_sts, y_valid_sts, y_test_sts = np.split(y_sts, split)

    df["Y"]["train"] = y_train_sts
    df["Y"]["valid"] = y_valid_sts
    df["Y"]["test"] = y_test_sts
    dataset["sequence"] = df

    if print_result:
        for df_form, ds in dataset.items():
            print(df_form, ":")
            for xy_label, xy in ds.items():
                print("  ", xy_label, ":")
                for split_name, split in xy.items():
                    print("    ", split_name, ": ", split.shape)

    return dataset
