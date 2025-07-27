import os.path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from variables import DATA_DIR, META_MS, META_HC, HC_FOLDER, MS_FOLDER, \
    MS_MAIN_TYPE, MS_TYPES


def read_excel_to_df(path: str, file_name: str) -> pd.DataFrame:
    """
    Loads xlsx from a given path
    :return: df loaded from path + filename
    """
    meta_df = pd.read_excel(os.path.join(path, file_name))
    return meta_df


def read_csv_to_df(path: str, file_name: str) -> pd.DataFrame:
    """
    Loads csv from a given path
    :return: df loaded from path + filename
    """
    meta_df = pd.read_csv(os.path.join(path, file_name), sep=";")
    return meta_df


def init_prepare_dfs(ms_dir, hc_dir):
    """
    Load initial ms data xls
    """
    # MS dataframe
    ms_df = read_excel_to_df(DATA_DIR, META_MS).drop(columns=["No."])
    ms_df = ms_df.rename(columns={
        "patID": "file_name", "Alter": "age", "Sex": "sex", "Disease Course/ Type at MRI date": "label"
    })
    ms_df["file_name"] = ms_df["file_name"].apply(lambda file_name: os.path.join(ms_dir, file_name))

    # HC dataframe
    hc_df = read_excel_to_df(DATA_DIR, META_HC)
    hc_df["Disease Course/ Type at MRI date"] = "HC"
    hc_df = hc_df.rename(columns={
        "subID": "file_name", "Alter": "age", "Sex": "sex", "Disease Course/ Type at MRI date": "label"
    })
    hc_df["file_name"] = hc_df["file_name"].apply(lambda file_name: os.path.join(hc_dir, file_name))

    merged_df = ms_df.merge(hc_df, on=['file_name', 'sex', 'age', 'label'], how='outer').sort_values(by=['file_name'])
    merged_df["file_name"] = merged_df["file_name"].apply(lambda file_name: file_name + ".nii.gz")
    return merged_df


def create_and_save_ms_df_base(ms_dir, hc_dir, file_name: str):
    """
    Create and save base csv with all paths and meta information of all data samples
    """
    assert hc_dir is not None and ms_dir is not None, "ms_dir and hc_dir must be given"
    full_df = init_prepare_dfs(ms_dir, hc_dir)
    full_df.to_csv(os.path.join(DATA_DIR, file_name), sep=";", index=False)


def create_training_splits(
        full_df,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        seed=None,
        subtypes: dict[str, int] = MS_MAIN_TYPE,
):
    """
    Create training, validation and test splits from the data
    :param subtypes: 0 - All, 1 - HC vs MS, 2 - MS_TYPES_FILTERED
    """
    train_df = pd.DataFrame() if train_frac is not None else None
    val_df = pd.DataFrame() if val_frac is not None else None
    test_df = pd.DataFrame() if test_frac is not None else None

    def sample_with_remove(df, frac: float, size: int, seed=None):
        """
        Sample the fraction of the size from the dataframe based on a seed
        """
        # either use fraction or take all remaining
        sample_size = min(round(size * frac), len(df))
        df_samples = df.sample(n=sample_size, random_state=seed)
        df_mod = df.drop(df_samples.index)
        return df_samples, df_mod

    for (ms_type, group_df) in full_df.groupby(by=["label"]):
        size = len(group_df)
        # use number encoding for ms_types
        if subtypes is not None:
            key = ms_type[0]
            if key not in subtypes:
                continue
            group_df["label"] = subtypes[key]
            group_df["initial_label"] = MS_TYPES[key]
        # order is important (fractions are rounded, therefore, first test is filled, then val and then train with
        # remaining)
        if test_frac:
            samples, group_df = sample_with_remove(group_df, test_frac, size, seed)
            test_df = pd.concat([test_df, samples], ignore_index=True)
        if val_frac:
            samples, group_df = sample_with_remove(group_df, val_frac, size, seed)
            val_df = pd.concat([val_df, samples], ignore_index=True)
        if train_frac:
            samples, group_df = sample_with_remove(group_df, train_frac, size, seed)
            train_df = pd.concat([train_df, samples], ignore_index=True)

    return train_df, val_df, test_df


def create_k_folds(
        full_df: pd.DataFrame,
        k: int = 10,
        shuffle: bool = True,
        seed: int =None,
        subtypes: dict[str, int] = MS_MAIN_TYPE,
):
    """
    Create k fold splits for performing k-fold cross validation

    :param shuffle:
    :param full_df:
    :param k:
    :param seed:
    :param subtypes:
    :return:
    """
    filtered_df = full_df[full_df['label'].isin(subtypes.keys())]

    for subtype in subtypes.keys():
        filtered_df.loc[filtered_df['label'] == subtype, 'label'] = subtypes[subtype]

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
    split_dfs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(filtered_df, filtered_df.label.astype(int))):
        train_df, test_df = filtered_df.iloc[train_idx], filtered_df.iloc[test_idx]
        split_dfs.append((train_df, test_df, None))
    return split_dfs


def save_splits(train_split: pd.DataFrame = None, val_split: pd.DataFrame = None, test_split: pd.DataFrame = None,
                base_name: str = "split", extra_path: str = ""):
    """
    Save splits to csv
    :return:
    """
    print(f"\nCreated splits for {base_name}:")
    if extra_path != "":
        os.makedirs(os.path.join(DATA_DIR, extra_path), exist_ok=True)
    if train_split is not None:
        if len(train_split) > 0: train_split.to_csv(os.path.join(DATA_DIR, os.path.join(extra_path, 'train_' + base_name + '.csv')), sep=";", index=False)
        print(f"train length: {len(train_split) if train_split is not None else '0'}")
    if val_split is not None:
        if len(val_split) > 0: val_split.to_csv(os.path.join(DATA_DIR, os.path.join(extra_path, 'val_' + base_name + '.csv')), sep=";", index=False)
        print(f"val length: {len(val_split) if val_split is not None else '0'}")
    if test_split is not None:
        if len(test_split) > 0: test_split.to_csv(os.path.join(DATA_DIR, os.path.join(extra_path, 'test_' + base_name + '.csv')), sep=";", index=False)
        print(f"test length: {len(test_split) if test_split is not None else '0'}")


if __name__ == "__main__":
    compute_full_dataset = False
    k_fold = False

    if compute_full_dataset:
        create_and_save_ms_df_base(MS_FOLDER, HC_FOLDER, 'full_data_original_size.csv')
    data = read_csv_to_df(DATA_DIR, "full_data_original_size.csv")
    if k_fold:
        splits = create_k_folds(data, 10, True, 0, MS_MAIN_TYPE)
        for i, split in enumerate(splits):
            save_splits(split[0], split[1], split[2], f"{i}-fold", "kFold_splits")
    else:
        # add subtypes = True, to get training splits with subtype labels instead of only HC and MS
        train, val, test = create_training_splits(data, seed=0, subtypes=MS_MAIN_TYPE, train_frac=0.7, val_frac=0.15,
                                                  test_frac=0.15)

        save_splits(train, val, test, base_name="split_tune_70")
