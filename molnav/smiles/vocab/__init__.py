"""SMILES Vocabulary Module.

This module provides utilities for loading and managing molecular vocabularies
used in SMILES tokenization and processing. It includes functions for loading
vocabulary files and defines standard tokens used in molecular machine learning tasks.

The vocabulary system is designed to support token-based representations of molecular
structures, enabling consistent encoding and decoding of SMILES strings for downstream
tasks such as molecular property prediction, generation, and analysis.

Functions
---------
load_vocab_text_file : callable
    Load a vocabulary from a text file and return as dict or DataFrame.

Provided Vocabulary Files
--------------------------
vocab_molnav_tok_base.txt
    Base vocabulary file containing standard molecular tokens used in SMILES strings.
    This vocabulary includes common atoms, bonds, brackets, and special tokens needed
    for tokenizing molecular representations. It serves as the foundation for molecular
    tokenization in the molnav package.

Standard Tokens
---------------
The module defines several special tokens used in molecular sequence processing:
- [CLS]: Classification token, typically used at the start of a sequence
- [EDGE]: Edge token for graph-based molecular representations
- [PAD]: Padding token for batch processing of variable-length sequences
- [MASK]: Mask token for masked language modeling tasks
- [SEP]: Separator token for distinguishing between sequence segments

See Also
--------
molnav.smiles.preprocessing : SMILES string preprocessing utilities
"""

from .load_vocab import load_vocab_text_file

__all__ = ["load_vocab_text_file"]
