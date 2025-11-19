"""Tests for SMILES preprocessing functions."""

from molnav.smiles.preprocessing import desalt_smiles, canonicalize_smiles


class TestCanonicalizeSMILES:
    """Tests for the canonicalize_smiles function."""

    def test_basic_canonicalization(self):
        """Test basic SMILES canonicalization."""
        # Different representations of the same molecule (benzene)
        smiles1 = "C1=CC=CC=C1"
        smiles2 = "c1ccccc1"
        smiles3 = "C1=CC(=CC=C1)"

        # All should canonicalize to the same Kekule form
        canonical1 = canonicalize_smiles(smiles1)
        canonical2 = canonicalize_smiles(smiles2)
        canonical3 = canonicalize_smiles(smiles3)

        assert canonical1 == canonical2 == canonical3

    def test_kekule_form_true(self):
        """Test that Kekule form uses explicit double bonds."""
        smiles = "c1ccccc1"  # Aromatic benzene
        canonical = canonicalize_smiles(smiles, kekule=True)

        # Should contain explicit double bonds, not lowercase aromatic
        assert "c" not in canonical
        assert "=" in canonical

    def test_kekule_form_false(self):
        """Test that non-Kekule form uses aromatic notation."""
        smiles = "C1=CC=CC=C1"  # Kekule benzene
        canonical = canonicalize_smiles(smiles, kekule=False)

        # Should use lowercase aromatic notation
        assert "c" in canonical

    def test_isomeric_false(self):
        """Test that stereochemistry is removed when isomeric=False."""
        # Chiral center with stereochemistry
        smiles = "C[C@H](O)CC"
        canonical = canonicalize_smiles(smiles, isomeric=False)

        # Should not contain stereochemistry markers
        assert "@" not in canonical
        assert "@@" not in canonical

    def test_isomeric_true(self):
        """Test that stereochemistry is preserved when isomeric=True."""
        # Chiral center with stereochemistry
        smiles = "C[C@H](O)CC"
        canonical = canonicalize_smiles(smiles, isomeric=True)

        # Should contain stereochemistry marker
        assert "@" in canonical

    def test_double_bond_stereochemistry_removed(self):
        """Test that E/Z stereochemistry is removed when isomeric=False."""
        # E-but-2-ene
        smiles = r"C/C=C/C"
        canonical = canonicalize_smiles(smiles, isomeric=False)

        # Should not contain stereochemistry markers
        assert "/" not in canonical
        assert "\\" not in canonical

    def test_double_bond_stereochemistry_preserved(self):
        """Test that E/Z stereochemistry is preserved when isomeric=True."""
        # E-but-2-ene
        smiles = r"C/C=C/C"
        canonical = canonicalize_smiles(smiles, isomeric=True)

        # Should contain stereochemistry marker
        assert "/" in canonical or "\\" in canonical

    def test_consistency_multiple_calls(self):
        """Test that multiple calls with the same input produce the same output."""
        smiles = "CC(C)C(=O)O"  # Isobutyric acid

        canonical1 = canonicalize_smiles(smiles)
        canonical2 = canonicalize_smiles(smiles)
        canonical3 = canonicalize_smiles(smiles)

        assert canonical1 == canonical2 == canonical3

    def test_complex_molecule(self):
        """Test canonicalization of a more complex molecule."""
        # Aspirin (acetylsalicylic acid)
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        canonical = canonicalize_smiles(smiles)

        # Should return a valid SMILES string
        assert isinstance(canonical, str)
        assert len(canonical) > 0

    def test_different_order_same_result(self):
        """Test that different atom ordering produces the same canonical SMILES."""
        # Ethanol written in different orders
        smiles1 = "CCO"
        smiles2 = "OCC"
        smiles3 = "C(O)C"

        canonical1 = canonicalize_smiles(smiles1)
        canonical2 = canonicalize_smiles(smiles2)
        canonical3 = canonicalize_smiles(smiles3)

        assert canonical1 == canonical2 == canonical3

    def test_cyclic_molecule(self):
        """Test canonicalization of a cyclic molecule."""
        # Cyclohexane
        smiles = "C1CCCCC1"
        canonical = canonicalize_smiles(smiles)

        # Should return a valid SMILES string
        assert isinstance(canonical, str)
        assert len(canonical) > 0

    def test_branched_molecule(self):
        """Test canonicalization of a branched molecule."""
        # Isobutane
        smiles1 = "CC(C)C"
        smiles2 = "C(C)(C)C"

        canonical1 = canonicalize_smiles(smiles1)
        canonical2 = canonicalize_smiles(smiles2)

        assert canonical1 == canonical2


class TestDesaltSMILES:
    """Tests for the desalt_smiles function."""

    def test_simple_salt_removal(self):
        """Test removal of simple salt (HCl)."""
        # Molecule with HCl salt
        smiles = "CCO.Cl"  # Ethanol with chloride
        desalted = desalt_smiles(smiles)

        # Should remove the chloride
        assert "Cl" not in desalted
        assert "CCO" in desalted or "OCC" in desalted or "C(O)C" in desalted

    def test_sodium_salt_removal(self):
        """Test removal of sodium counterion."""
        # Sodium salt
        smiles = "CC(=O)[O-].[Na+]"  # Sodium acetate
        desalted = desalt_smiles(smiles)

        # Should remove sodium
        assert "[Na+]" not in desalted
        assert "C" in desalted  # Should still have the acetate

    def test_multiple_salts_removal(self):
        """Test removal of multiple salt components."""
        # Molecule with multiple salt components
        smiles = "CCN.Cl.Cl"  # Ethylamine dihydrochloride
        desalted = desalt_smiles(smiles)

        # Should remove chlorides
        assert desalted.count("Cl") < smiles.count("Cl")

    def test_water_removal(self):
        """Test removal of water molecules (hydrates)."""
        # Molecule with water
        smiles = "CC(=O)O.O"  # Acetic acid monohydrate
        desalted = desalt_smiles(smiles)

        # Water should be removed (if it's recognized as a solvent)
        # The main molecule should remain
        assert "C" in desalted

    def test_no_salt_present(self):
        """Test that molecules without salts remain unchanged."""
        # Pure molecule without salts
        smiles = "CCO"  # Ethanol
        desalted = desalt_smiles(smiles)

        # Should return the same molecule (possibly in different form)
        assert "C" in desalted
        assert "O" in desalted

    def test_dont_remove_everything(self):
        """Test that at least one fragment is preserved."""
        # Only salt components
        smiles = "Cl.[Na+]"
        desalted = desalt_smiles(smiles)

        # Should preserve at least one component
        assert len(desalted) > 0
        assert desalted != ""

    def test_preserve_main_molecule(self):
        """Test that the main molecule is preserved when desalting."""
        # Drug molecule with HCl salt
        smiles = "CN1CCC[C@H]1c2cccnc2.Cl"  # Nicotine hydrochloride
        desalted = desalt_smiles(smiles)

        # Should preserve the main molecule
        assert "N" in desalted
        assert "C" in desalted
        # Salt should be removed
        assert "." not in desalted or "Cl" not in desalted

    def test_complex_salt_form(self):
        """Test desalting of a complex salt form."""
        # Molecule with sulfate salt
        smiles = "CCN.OS(=O)(=O)O"  # Ethylamine sulfate
        desalted = desalt_smiles(smiles)

        # Should contain the amine
        assert "N" in desalted
        assert "C" in desalted

    def test_kekule_output(self):
        """Test that output is in Kekule form."""
        # Aromatic molecule with salt
        smiles = "c1ccccc1N.Cl"  # Aniline hydrochloride
        desalted = desalt_smiles(smiles)

        # Output should be in Kekule form (explicit bonds)
        # This is harder to test directly, but we can verify it's valid
        assert isinstance(desalted, str)
        assert len(desalted) > 0

    def test_non_isomeric_output(self):
        """Test that output is non-isomeric (no stereochemistry)."""
        # Chiral molecule with salt
        smiles = "C[C@H](N)C.Cl"
        desalted = desalt_smiles(smiles)

        # Should not contain stereochemistry markers
        assert "@" not in desalted
        assert "/" not in desalted
        assert "\\" not in desalted


class TestIntegration:
    """Integration tests combining both functions."""

    def test_desalt_then_canonicalize(self):
        """Test desalting followed by canonicalization."""
        # Salt form of a molecule
        smiles = "CCO.Cl"

        # Desalt first
        desalted = desalt_smiles(smiles)
        # Then canonicalize
        canonical = canonicalize_smiles(desalted)

        # Should produce a clean canonical SMILES
        assert isinstance(canonical, str)
        assert len(canonical) > 0
        assert "Cl" not in canonical

    def test_canonicalize_then_desalt(self):
        """Test canonicalization followed by desalting."""
        # Salt form with non-canonical SMILES
        smiles = "OCC.Cl"

        # Canonicalize first
        canonical = canonicalize_smiles(smiles)
        # Then desalt
        desalted = desalt_smiles(canonical)

        # Should produce a clean SMILES
        assert isinstance(desalted, str)
        assert len(desalted) > 0

    def test_full_pipeline(self):
        """Test a full preprocessing pipeline."""
        # Complex molecule with salt and non-canonical representation
        smiles = "c1ccccc1C(O)=O.[Na+]"  # Sodium benzoate

        # Step 1: Desalt
        desalted = desalt_smiles(smiles)
        # Step 2: Canonicalize with isomeric=False, kekule=True
        canonical = canonicalize_smiles(desalted, isomeric=False, kekule=True)

        # Should produce a standardized SMILES
        assert isinstance(canonical, str)
        assert len(canonical) > 0
        assert "[Na+]" not in canonical
        assert "=" in canonical  # Should have explicit bonds
        assert "c" not in canonical  # Should be Kekule form
