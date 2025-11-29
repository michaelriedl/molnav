"""Standard Tokens for Molecular Sequence Processing.

This module defines special tokens used in molecular sequence processing and
tokenization tasks. These tokens are essential for various machine learning
operations on molecular data, including sequence classification, masked language
modeling, and graph-based molecular representations.

Token Definitions
-----------------
CLS_TOKEN : str
    Classification token ("[CLS]"). Used at the beginning of a sequence to represent
    the entire sequence in classification tasks. The embedding of this token is
    typically used as the aggregate sequence representation for downstream tasks.

EDGE_TOKEN : str
    Edge token ("[EDGE]"). Used in graph-based molecular representations to denote
    connections or edges between molecular substructures or nodes. Can also be used
    to denote the start and end of a SMILES string.

PAD_TOKEN : str
    Padding token ("[PAD]"). Used to pad sequences to a uniform length when batching
    variable-length molecular sequences. This ensures all sequences in a batch have
    the same dimensions for efficient processing.

MASK_TOKEN : str
    Mask token ("[MASK]"). Used in masked language modeling tasks where certain tokens
    in a sequence are hidden and the model must predict the original tokens. This is
    commonly used for self-supervised pre-training of molecular models.

SEP_TOKEN : str
    Separator token ("[SEP]"). Used to separate different segments or sequences in
    multi-sequence tasks, such as when processing multiple molecules or different
    parts of a molecular structure. This is also used for self-supervised pre-training
    tasks like next sentence prediction.

Examples
--------
>>> from molnav.smiles.vocab.standard_tokens import CLS_TOKEN, PAD_TOKEN
>>> sequence = [CLS_TOKEN, "C", "C", "O", PAD_TOKEN, PAD_TOKEN]
>>> print(sequence)
['[CLS]', 'C', 'C', 'O', '[PAD]', '[PAD]']
"""

# Define the standard tokens
CLS_TOKEN = "[CLS]"
EDGE_TOKEN = "[EDGE]"
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
SEP_TOKEN = "[SEP]"
