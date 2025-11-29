import os

import pandas as pd


def load_vocab_text_file(vocab_file_name: str, return_type: str = "dict") -> dict | pd.DataFrame:
    """Load an existing vocabulary from a text file for tokenization purposes.

    This function reads a vocabulary file containing molecular tokens (such as atoms,
    bonds, or special tokens) and creates a mapping between tokens and their corresponding
    indices. The vocabulary file should contain one token per line, with each line
    representing a unique token in the vocabulary. The function assigns sequential integer
    indices starting from 0 based on the order of tokens in the file.

    The function supports two output formats: a dictionary mapping tokens to indices (suitable
    for quick lookups during tokenization) or a pandas DataFrame (useful for analysis,
    inspection, or integration with data processing pipelines).

    Args:
        vocab_file_name: The file path to the vocabulary text file to be loaded. This should
            be a plain text file with one token per line. Each line will be read as a separate
            vocabulary token, and the order of lines determines the index assignment. The file
            should not contain a header row.
        return_type: The format in which to return the vocabulary data. Options are 'dict'
            (default) or 'dataframe'. When set to 'dict', returns a dictionary with tokens as
            keys and their integer indices as values, providing O(1) lookup time for tokenization
            operations. When set to 'dataframe', returns a pandas DataFrame with tokens as the
            index and a single 'idx' column containing the integer indices, which is useful for
            data exploration and analysis. Defaults to 'dict'.

    Returns:
        If `return_type` is 'dict', returns a dictionary where each key is a token string from
        the vocabulary file and each value is the corresponding integer index (0-based, assigned
        sequentially based on file order). If `return_type` is 'dataframe', returns a pandas
        DataFrame with tokens as the DataFrame index and an 'idx' column containing the
        corresponding integer indices.

    Raises:
        FileNotFoundError: If the specified vocabulary file does not exist.
        ValueError: If the `return_type` parameter is not 'dict' or 'dataframe'.
    """
    # Check if file exists
    if not os.path.isfile(vocab_file_name):
        raise FileNotFoundError(f"The vocabulary file '{vocab_file_name}' does not exist.")
    # Check the return type
    if return_type not in ["dict", "dataframe"]:
        raise ValueError("Invalid return type. Must be 'dict' or 'dataframe'.")

    # Get vocabulary
    vocab = pd.read_csv(vocab_file_name, header=None)[0].to_list()
    vocab_data = {v: ind for ind, v in enumerate(vocab)}
    # Convert to DataFrame if needed
    if return_type == "dataframe":
        vocab_data = pd.DataFrame(vocab_data.values(), columns=["idx"], index=vocab_data.keys())

    return vocab_data
