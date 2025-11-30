import re

import torch
import pandas as pd

from .vocab.standard_tokens import CLS_TOKEN, PAD_TOKEN, EDGE_TOKEN


def smiles_tokenizer(smiles: str) -> list:
    """Tokenize a SMILES string into its constituent molecular components.

    This function breaks down a SMILES (Simplified Molecular Input Line Entry System) string
    into individual tokens representing atoms, bonds, rings, branches, and other structural
    features of a molecule. The tokenization follows the pattern used in the Molecular
    Transformer architecture (Schwaller et al., 2019), which is specifically designed to
    capture the syntactic elements of SMILES notation.

    The tokenizer recognizes various SMILES components including:
    - Atoms: Single-letter (C, N, O, etc.) and two-letter (Br, Cl) atoms
    - Aromatic atoms: Lowercase letters (c, n, o, s, p)
    - Brackets: Square brackets containing atom specifications with charges, isotopes, etc.
    - Bonds: Single (implicit or -), double (=), triple (#), aromatic (:)
    - Branches: Parentheses for branching points
    - Rings: Numeric ring closure indicators and %XX for rings > 9
    - Stereochemistry: /, \, @, @@ symbols
    - Other special characters: ., +, -, >, *, $, ~, ?

    Args:
        smiles: The input SMILES string to be tokenized. This should be a valid SMILES
            representation of a molecular structure. The function will validate that all
            characters in the input conform to recognized SMILES tokens.

    Returns:
        A list of string tokens extracted from the input SMILES string. Each token represents
        a single structural element (atom, bond, branch marker, etc.) in the order they appear
        in the SMILES string. The tokens preserve the original SMILES syntax and can be
        concatenated to reconstruct the original string.

    Raises:
        ValueError: If the input SMILES string contains characters that are not recognized
            by the tokenization pattern. This indicates the SMILES string may be malformed
            or contain invalid characters not part of standard SMILES notation.
    """
    # Tokenization pattern from Molecular Transformer - Schwaller et al., 2019
    pattern = (
        "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
        "\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smiles)]
    # Check if the smiles string had extra characters not recognized by regex.
    # Solution based on https://stackoverflow.com/a/3879574
    if len("".join(tokens)) < len(smiles):
        raise ValueError("Input smiles string contained invalid characters.")

    return tokens


def smiles_to_tensor(
    smiles: str,
    vocab: dict | pd.DataFrame,
    tokenizer: callable = smiles_tokenizer,
    max_seq_len: int | None = None,
    add_edge_tokens=True,
    add_cls_token=True,
) -> torch.Tensor:
    """Convert a SMILES string to a PyTorch tensor using vocabulary-based token encoding.

    This function transforms a SMILES string into a numerical tensor representation suitable
    for input to deep learning models. The conversion process involves several steps: tokenizing
    the SMILES string into constituent elements, mapping each token to its integer index in the
    provided vocabulary, optionally adding special tokens (classification and edge markers), and
    padding the sequence to a specified maximum length.

    The function supports transformer-based architectures by providing options to add [CLS] tokens
    (commonly used for classification tasks) and [EDGE] tokens (which mark sequence boundaries).
    Sequences shorter than the maximum length are right-padded with [PAD] tokens to ensure uniform
    tensor dimensions across a batch of molecules.

    The resulting tensor can be directly fed into neural network models for molecular property
    prediction, generative modeling, or other machine learning tasks involving molecular structures.

    Args:
        smiles: The input SMILES string representing the molecular structure to be converted.
            This should be a valid SMILES string that can be successfully tokenized by the
            provided tokenizer function.
        vocab: The vocabulary mapping tokens to integer indices. Can be provided as either a
            dictionary where keys are token strings and values are integer indices, or as a
            pandas DataFrame where the index contains tokens and an 'idx' column contains the
            corresponding integer indices. All tokens in the SMILES string (after tokenization)
            must exist in this vocabulary, along with special tokens [PAD], [CLS], and [EDGE]
            if the corresponding flags are enabled.
        tokenizer: The tokenization function to use for breaking down the SMILES string into
            tokens. This should be a callable that accepts a SMILES string and returns a list
            of token strings. By default, uses the `smiles_tokenizer` function from this module,
            which implements the Molecular Transformer tokenization pattern. Defaults to
            `smiles_tokenizer`.
        max_seq_len: The maximum sequence length for the output tensor. If specified, sequences
            shorter than this length will be right-padded with [PAD] tokens to reach the maximum
            length. Sequences longer than this length will NOT be truncated and will retain their
            full length. If set to None, the maximum sequence length equals the actual length of
            the tokenized and augmented SMILES string (including any added special tokens), with
            no padding applied. Defaults to None.
        add_edge_tokens: A flag controlling whether to add [EDGE] tokens at the beginning and
            end of the tokenized sequence. Edge tokens serve as explicit boundary markers that
            can help transformer models distinguish between padding and actual sequence boundaries.
            When enabled, one [EDGE] token is prepended and one is appended to the tokenized
            sequence. Defaults to True.
        add_cls_token: A flag controlling whether to add a [CLS] (classification) token at the
            very beginning of the sequence. This token is commonly used in transformer architectures
            (such as BERT) as a special token whose learned representation captures information
            about the entire sequence, making it useful for sequence-level prediction tasks.
            The [CLS] token, if added, appears before any [EDGE] tokens. Defaults to True.

    Returns:
        A PyTorch tensor of dtype `torch.long` containing the integer-encoded representation
        of the SMILES string. The tensor has shape (1, L) where L is either `max_seq_len`
        (if specified and greater than the sequence length) or the actual length of the
        tokenized sequence including any added special tokens. The first dimension (batch
        dimension) is always 1. Values in the tensor correspond to vocabulary indices, with
        padding positions filled with the [PAD] token index. The tensor is created with
        `requires_grad=False` as it represents discrete token indices rather than learnable
        parameters.
    """
    # Convert the DataFrame to a dictionary
    if isinstance(vocab, pd.DataFrame):
        vocab = vocab["idx"].to_dict()
    # Tokenize the SMILES
    smiles_tok = tokenizer(smiles)
    tok = [vocab[x] for x in smiles_tok]
    # Add in the edge tokens
    if add_edge_tokens:
        tok = [vocab[EDGE_TOKEN], *tok, vocab[EDGE_TOKEN]]
    # Add in the classification token
    if add_cls_token:
        tok = [vocab[CLS_TOKEN], *tok]
    # Combine the tokens into a tensor
    smiles_ten = torch.tensor(tok, dtype=torch.long, requires_grad=False)
    if max_seq_len is None:
        max_seq_len = len(smiles_ten)
    smiles_ten_long = torch.full((1, max_seq_len), vocab[PAD_TOKEN], dtype=torch.long)
    smiles_ten_long[0, : smiles_ten.shape[0]] = smiles_ten

    return smiles_ten_long
