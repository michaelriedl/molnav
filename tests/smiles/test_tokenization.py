"""Tests for SMILES tokenization functions."""

import os

import torch
import pandas as pd
import pytest

from molnav.smiles.tokenization import smiles_to_tensor, smiles_tokenizer
from molnav.smiles.vocab.load_vocab import load_vocab_text_file
from molnav.smiles.vocab.standard_tokens import (
    CLS_TOKEN,
    PAD_TOKEN,
    EDGE_TOKEN,
    MASK_TOKEN,
)


class TestSmilesTokenizer:
    """Tests for the smiles_tokenizer function."""

    def test_simple_molecule(self):
        """Test tokenization of a simple molecule (ethanol)."""
        smiles = "CCO"
        tokens = smiles_tokenizer(smiles)

        assert tokens == ["C", "C", "O"]
        assert len(tokens) == 3

    def test_benzene_aromatic(self):
        """Test tokenization of aromatic benzene."""
        smiles = "c1ccccc1"
        tokens = smiles_tokenizer(smiles)

        # Should tokenize as individual aromatic carbons and ring numbers
        assert tokens == ["c", "1", "c", "c", "c", "c", "c", "1"]
        assert len(tokens) == 8

    def test_branched_molecule(self):
        """Test tokenization of a branched molecule."""
        smiles = "CC(C)C"
        tokens = smiles_tokenizer(smiles)

        # Should preserve parentheses for branches
        assert "(" in tokens
        assert ")" in tokens
        assert tokens == ["C", "C", "(", "C", ")", "C"]

    def test_double_bond(self):
        """Test tokenization of double bond."""
        smiles = "C=C"
        tokens = smiles_tokenizer(smiles)

        assert "=" in tokens
        assert tokens == ["C", "=", "C"]

    def test_triple_bond(self):
        """Test tokenization of triple bond."""
        smiles = "C#N"
        tokens = smiles_tokenizer(smiles)

        assert "#" in tokens
        assert tokens == ["C", "#", "N"]

    def test_aromatic_nitrogen(self):
        """Test tokenization of aromatic nitrogen."""
        smiles = "c1ncccc1"
        tokens = smiles_tokenizer(smiles)

        assert "n" in tokens
        assert tokens == ["c", "1", "n", "c", "c", "c", "c", "1"]

    def test_two_letter_atoms(self):
        """Test tokenization of two-letter atoms (Cl, Br)."""
        smiles = "CCCl"
        tokens = smiles_tokenizer(smiles)

        # Cl should be tokenized as a single token
        assert "Cl" in tokens
        assert tokens == ["C", "C", "Cl"]

        smiles = "CCBr"
        tokens = smiles_tokenizer(smiles)

        # Br should be tokenized as a single token
        assert "Br" in tokens
        assert tokens == ["C", "C", "Br"]

    def test_brackets_with_charges(self):
        """Test tokenization of brackets with charges."""
        smiles = "CC[O-]"
        tokens = smiles_tokenizer(smiles)

        # Bracket expression should be a single token
        assert "[O-]" in tokens
        assert tokens == ["C", "C", "[O-]"]

    def test_brackets_with_isotopes(self):
        """Test tokenization of isotopes in brackets."""
        smiles = "[13C]C"
        tokens = smiles_tokenizer(smiles)

        # Isotope in brackets should be a single token
        assert "[13C]" in tokens
        assert tokens == ["[13C]", "C"]

    def test_stereochemistry_chiral(self):
        """Test tokenization of chiral centers."""
        smiles = "C[C@H](O)C"
        tokens = smiles_tokenizer(smiles)

        # Should preserve stereochemistry marker
        assert "[C@H]" in tokens
        assert "@" not in tokens  # @ should be inside the bracket

    def test_stereochemistry_double_bond(self):
        """Test tokenization of E/Z stereochemistry."""
        smiles = r"C/C=C/C"
        tokens = smiles_tokenizer(smiles)

        # Should preserve / and \ markers
        assert "/" in tokens
        assert tokens.count("/") == 2

    def test_ring_closures_single_digit(self):
        """Test tokenization of single-digit ring closures."""
        smiles = "C1CCCCC1"
        tokens = smiles_tokenizer(smiles)

        # Ring numbers should be separate tokens
        assert "1" in tokens
        assert tokens.count("1") == 2

    def test_ring_closures_two_digit(self):
        """Test tokenization of two-digit ring closures."""
        smiles = "C%10CCCCC%10"
        tokens = smiles_tokenizer(smiles)

        # Two-digit ring numbers should be single tokens with %
        assert "%10" in tokens
        assert tokens.count("%10") == 2

    def test_disconnected_fragments(self):
        """Test tokenization of disconnected fragments."""
        smiles = "CCO.Cl"
        tokens = smiles_tokenizer(smiles)

        # Dot should be preserved as separator
        assert "." in tokens
        assert tokens == ["C", "C", "O", ".", "Cl"]

    def test_complex_molecule_aspirin(self):
        """Test tokenization of a complex molecule (aspirin)."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        tokens = smiles_tokenizer(smiles)

        # Should be a list of tokens
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Check some expected tokens
        assert "C" in tokens
        assert "O" in tokens
        assert "=" in tokens
        assert "(" in tokens
        assert ")" in tokens
        assert "c" in tokens

    def test_empty_string_returns_empty_list(self):
        """Test that empty string returns empty list."""
        smiles = ""
        tokens = smiles_tokenizer(smiles)

        assert tokens == []

    def test_invalid_characters_raise_error(self):
        """Test that invalid characters raise ValueError."""
        # SMILES with invalid characters
        smiles = "CC&O"  # & is not a valid SMILES character

        with pytest.raises(ValueError, match="invalid characters"):
            smiles_tokenizer(smiles)

    def test_tokenization_preserves_order(self):
        """Test that tokenization preserves the order of atoms."""
        smiles = "CCCCCCCCCC"
        tokens = smiles_tokenizer(smiles)

        # Should be 10 C tokens in order
        assert tokens == ["C"] * 10

    def test_concatenation_equals_original(self):
        """Test that concatenating tokens reproduces original SMILES."""
        smiles = "c1ccccc1C(=O)O"
        tokens = smiles_tokenizer(smiles)

        # Joining tokens should give back the original SMILES
        assert "".join(tokens) == smiles


class TestSmilesToTensor:
    """Tests for the smiles_to_tensor function."""

    @pytest.fixture
    def simple_vocab(self):
        """Create a simple vocabulary for testing."""
        return {
            PAD_TOKEN: 0,
            CLS_TOKEN: 1,
            EDGE_TOKEN: 2,
            MASK_TOKEN: 3,
            "C": 4,
            "O": 5,
            "N": 6,
            "=": 7,
            "(": 8,
            ")": 9,
            "c": 10,
            "1": 11,
        }

    def test_basic_conversion(self, simple_vocab):
        """Test basic SMILES to tensor conversion."""
        smiles = "CCO"
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            add_edge_tokens=False,
            add_cls_token=False,
        )

        # Should be a 2D tensor
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 2
        assert tensor.shape[0] == 1

        # Should contain the correct token indices
        assert tensor[0, 0] == simple_vocab["C"]
        assert tensor[0, 1] == simple_vocab["C"]
        assert tensor[0, 2] == simple_vocab["O"]

    def test_with_cls_token(self, simple_vocab):
        """Test adding CLS token at the beginning."""
        smiles = "CCO"
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            add_edge_tokens=False,
            add_cls_token=True,
        )

        # First token should be CLS
        assert tensor[0, 0] == simple_vocab[CLS_TOKEN]
        # Following tokens should be the molecule
        assert tensor[0, 1] == simple_vocab["C"]
        assert tensor[0, 2] == simple_vocab["C"]
        assert tensor[0, 3] == simple_vocab["O"]

    def test_with_edge_tokens(self, simple_vocab):
        """Test adding EDGE tokens at boundaries."""
        smiles = "CCO"
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            add_edge_tokens=True,
            add_cls_token=False,
        )

        # First token should be EDGE
        assert tensor[0, 0] == simple_vocab[EDGE_TOKEN]
        # Last non-pad token should be EDGE
        # EDGE + C + C + O + EDGE = 5 tokens
        assert tensor[0, 4] == simple_vocab[EDGE_TOKEN]

    def test_with_both_cls_and_edge(self, simple_vocab):
        """Test adding both CLS and EDGE tokens."""
        smiles = "CCO"
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            add_edge_tokens=True,
            add_cls_token=True,
        )

        # First token should be CLS
        assert tensor[0, 0] == simple_vocab[CLS_TOKEN]
        # Second token should be EDGE
        assert tensor[0, 1] == simple_vocab[EDGE_TOKEN]
        # Last non-pad token should be EDGE
        # CLS + EDGE + C + C + O + EDGE = 6 tokens
        assert tensor[0, 5] == simple_vocab[EDGE_TOKEN]

    def test_without_special_tokens(self, simple_vocab):
        """Test conversion without any special tokens."""
        smiles = "CCO"
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            add_edge_tokens=False,
            add_cls_token=False,
        )

        # Should only have the molecule tokens
        # C + C + O = 3 tokens
        assert tensor.shape[1] == 3
        assert tensor[0, 0] == simple_vocab["C"]

    def test_padding_to_max_length(self, simple_vocab):
        """Test that sequences are padded to max_seq_len."""
        smiles = "CCO"
        max_len = 10
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            max_seq_len=max_len,
            add_edge_tokens=False,
            add_cls_token=False,
        )

        # Should be padded to max_len
        assert tensor.shape[1] == max_len
        # Padding tokens should be at the end
        assert tensor[0, 3] == simple_vocab[PAD_TOKEN]
        assert tensor[0, 9] == simple_vocab[PAD_TOKEN]

    def test_no_padding_when_max_len_none(self, simple_vocab):
        """Test that no padding occurs when max_seq_len is None."""
        smiles = "CCO"
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            max_seq_len=None,
            add_edge_tokens=False,
            add_cls_token=False,
        )

        # Should match the exact length of the tokenized SMILES
        # C + C + O = 3 tokens
        assert tensor.shape[1] == 3

    def test_correct_sequence_length_calculations(self, simple_vocab):
        """Test that sequence length is calculated correctly with special tokens."""
        smiles = "CCO"  # 3 tokens

        # Without special tokens
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            add_edge_tokens=False,
            add_cls_token=False,
        )
        assert tensor.shape[1] == 3

        # With CLS only
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            add_edge_tokens=False,
            add_cls_token=True,
        )
        assert tensor.shape[1] == 4  # 3 + 1

        # With EDGE only
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            add_edge_tokens=True,
            add_cls_token=False,
        )
        assert tensor.shape[1] == 5  # 3 + 2

        # With both
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            add_edge_tokens=True,
            add_cls_token=True,
        )
        assert tensor.shape[1] == 6  # 3 + 1 + 2

    def test_dataframe_vocab_input(self, simple_vocab):
        """Test that DataFrame vocabulary input works correctly."""
        # Convert dict to DataFrame
        vocab_df = pd.DataFrame.from_dict(simple_vocab, orient="index", columns=["idx"])

        smiles = "CCO"
        tensor = smiles_to_tensor(
            smiles,
            vocab_df,
            add_edge_tokens=False,
            add_cls_token=False,
        )

        # Should produce the same result as dict input
        assert isinstance(tensor, torch.Tensor)
        assert tensor[0, 0] == simple_vocab["C"]
        assert tensor[0, 1] == simple_vocab["C"]
        assert tensor[0, 2] == simple_vocab["O"]

    def test_custom_tokenizer(self, simple_vocab):
        """Test using a custom tokenizer function."""

        # Simple custom tokenizer that splits on every character
        def custom_tokenizer(smiles):
            return list(smiles)

        # Add single character tokens to vocab
        vocab_extended = simple_vocab.copy()

        smiles = "CO"
        tensor = smiles_to_tensor(
            smiles,
            vocab_extended,
            tokenizer=custom_tokenizer,
            add_edge_tokens=False,
            add_cls_token=False,
        )

        # Should use custom tokenizer
        assert tensor.shape[1] == 2

    def test_tensor_dtype_and_grad(self, simple_vocab):
        """Test that tensor has correct dtype and gradient settings."""
        smiles = "CCO"
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            add_edge_tokens=False,
            add_cls_token=False,
        )

        # Should be long dtype
        assert tensor.dtype == torch.long
        # Should not require gradients
        assert not tensor.requires_grad

    def test_batch_dimension(self, simple_vocab):
        """Test that tensor has batch dimension."""
        smiles = "CCO"
        tensor = smiles_to_tensor(
            smiles,
            simple_vocab,
            add_edge_tokens=False,
            add_cls_token=False,
        )

        # First dimension should be 1 (batch size)
        assert tensor.shape[0] == 1

    def test_complex_smiles(self, simple_vocab):
        """Test conversion of a more complex SMILES."""
        # Extend vocab with needed tokens
        vocab_extended = simple_vocab.copy()
        vocab_extended.update(
            {
                "n": 12,
                "2": 13,
                "3": 14,
            }
        )

        smiles = "c1ncc(=O)cc1"
        tensor = smiles_to_tensor(
            smiles,
            vocab_extended,
            add_edge_tokens=False,
            add_cls_token=False,
        )

        # Should successfully convert
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 1
        assert tensor.shape[1] > 0


class TestIntegration:
    """Integration tests combining tokenization and tensor conversion."""

    def test_tokenize_then_tensorize(self):
        """Test the full pipeline from SMILES to tensor."""
        smiles = "CCO"
        vocab = {
            PAD_TOKEN: 0,
            CLS_TOKEN: 1,
            EDGE_TOKEN: 2,
            "C": 3,
            "O": 4,
        }

        # Step 1: Tokenize
        tokens = smiles_tokenizer(smiles)
        assert tokens == ["C", "C", "O"]

        # Step 2: Convert to tensor
        tensor = smiles_to_tensor(
            smiles,
            vocab,
            add_edge_tokens=True,
            add_cls_token=True,
        )

        # Should produce a valid tensor
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 1

    def test_load_vocab_and_tensorize(self):
        """Test loading vocabulary from file and tensorizing."""
        # Get the path to the vocabulary file
        vocab_path = os.path.join(
            os.path.dirname(__file__), "../../molnav/smiles/vocab/vocab_moltransformer_tok_base.txt"
        )

        # Check if vocab file exists
        if os.path.exists(vocab_path):
            # Load vocabulary
            vocab = load_vocab_text_file(vocab_path, return_type="dict")

            # Test conversion
            smiles = "CCO"
            tensor = smiles_to_tensor(
                smiles,
                vocab,
                add_edge_tokens=True,
                add_cls_token=True,
            )

            # Should produce a valid tensor
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape[0] == 1
            assert tensor[0, 0] == vocab[CLS_TOKEN]
            assert tensor[0, 1] == vocab[EDGE_TOKEN]


@pytest.mark.parametrize(
    ("add_edge_tokens", "add_cls_token"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_special_tokens_parametrized(add_edge_tokens: bool, add_cls_token: bool):
    """Test various combinations of special token flags."""
    smiles = "CCO"
    vocab = {
        PAD_TOKEN: 0,
        CLS_TOKEN: 1,
        EDGE_TOKEN: 2,
        "C": 3,
        "O": 4,
    }

    tensor = smiles_to_tensor(
        smiles,
        vocab,
        max_seq_len=None,
        add_edge_tokens=add_edge_tokens,
        add_cls_token=add_cls_token,
    )

    # Calculate expected length
    expected_len = 3  # Base tokens: C, C, O
    if add_edge_tokens:
        expected_len += 2  # EDGE at start and end
    if add_cls_token:
        expected_len += 1  # CLS at very start

    assert tensor.shape[1] == expected_len

    # Check token positions
    if add_cls_token:
        assert tensor[0, 0] == vocab[CLS_TOKEN]
    else:
        assert tensor[0, 0] != vocab[CLS_TOKEN]

    if add_edge_tokens:
        assert tensor[0, -1] == vocab[EDGE_TOKEN]
    else:
        assert tensor[0, -1] != vocab[EDGE_TOKEN]
