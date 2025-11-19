import numpy as np
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


def canonicalize_smiles(smiles: str, isomeric: bool = False, kekule: bool = False) -> str:
    """Canonicalize a SMILES string using RDKit's canonicalization algorithm.

    This function converts a SMILES (Simplified Molecular Input Line Entry System) string
    into its canonical form, ensuring that the same molecule always receives the same SMILES
    representation regardless of the input format. Canonicalization is essential for molecular
    comparison, database searching, and deduplication of chemical structures.

    The function provides options to control the representation of stereochemistry and
    aromaticity in the output SMILES string. By default, stereochemical information is
    removed and aromatic notation (lowercase letters for aromatic atoms) is used.

    Args:
        smiles: The input SMILES string representing the molecular structure to be canonicalized.
            This should be a valid SMILES string that RDKit can parse into a molecular object.
        isomeric: A flag controlling whether stereochemical information should be preserved
            in the canonical SMILES output. When set to False (default), all stereochemistry
            information (R/S chirality, E/Z double bond configuration) is removed from the
            output. When set to True, the canonical SMILES will include stereochemical
            descriptors (@, @@, /, \) where applicable. Defaults to False.
        kekule: A flag controlling the representation of aromatic systems in the output.
            When set to False (default), aromatic atoms are represented with lowercase letters
            (e.g., 'c' for aromatic carbon), providing a more compact representation. When set
            to True, aromatic rings are represented using alternating single and double bonds
            (Kekule structure) rather than lowercase aromatic atom symbols, which provides a
            more explicit representation of the bonding. Defaults to False.

    Returns:
        A canonicalized SMILES string representation of the input molecule. The output
        format depends on the `isomeric` and `kekule` parameters. This canonical form
        ensures consistent representation of the same molecular structure across different
        input variations.
    """
    # Convert the SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    # Convert back to SMILES
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric, kekuleSmiles=kekule)

    return smiles


def desalt_smiles(smiles: str) -> str:
    """Remove salts and solvents from a SMILES string to obtain the main molecular structure.

    This function identifies and removes common salts, counterions, and solvents from a SMILES
    string that may represent a salt form or solvate of a compound. In chemical databases and
    experimental data, molecules are often stored or reported with associated counterions
    (e.g., chloride, sodium) or solvent molecules (e.g., water, ethanol). This function uses
    RDKit's SaltRemover to strip these additional components and return only the main molecular
    entity of interest.

    The desalting process is crucial for standardizing molecular structures, as the same active
    pharmaceutical ingredient or chemical compound may appear in different salt forms. By removing
    salts, you can ensure consistent molecular representations for downstream analysis, comparison,
    or machine learning applications.

    Note that the function returns a non-canonical, non-isomeric SMILES string using aromatic
    notation. If you need a canonical representation, consider using `canonicalize_smiles()` on
    the output.

    Args:
        smiles: The input SMILES string that may contain salts, counterions, or solvent molecules
            in addition to the main molecular structure. This should be a valid SMILES string that
            RDKit can parse. Multiple molecular fragments in the SMILES (indicated by '.' separator)
            will be processed, with salt/solvent components removed according to RDKit's default
            salt definitions.

    Returns:
        A SMILES string representing the desalted molecular structure with salts and solvents
        removed. The output uses aromatic notation (lowercase letters for aromatic atoms) without
        stereochemical information. If the input consists entirely of salt/solvent molecules, the
        function will preserve at least one fragment to avoid returning an empty result (due to the
        `dontRemoveEverything=True` parameter).
    """
    # Initialize the salt/solvent remover
    remover = SaltRemover()
    # Convert the SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    # Remove the salts/solvents
    mol = remover.StripMol(mol, dontRemoveEverything=True)
    # Convert back to SMILES
    smiles = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False, kekuleSmiles=False)

    return smiles


def sanitize_smiles(smiles: str) -> str:
    """Sanitize a SMILES string by removing salts and producing a canonical representation.

    This function performs a two-step standardization process on a SMILES string to prepare
    it for downstream analysis, comparison, or storage. First, it removes any salts, counterions,
    and solvent molecules that may be present in the input SMILES, isolating the main molecular
    structure of interest. Second, it canonicalizes the desalted molecule to ensure a unique,
    standardized representation.

    Sanitization is a critical preprocessing step in cheminformatics workflows, particularly
    when working with molecules from diverse sources such as chemical databases, high-throughput
    screening data, or literature extraction. Different sources may report the same compound
    in various salt forms or with different SMILES notations. This function ensures that
    equivalent molecular structures receive identical SMILES representations, enabling accurate
    molecular comparison, deduplication, and consistent machine learning feature extraction.

    The function combines `desalt_smiles()` to remove extraneous molecular fragments with
    `canonicalize_smiles()` using default parameters (non-isomeric, non-Kekule) to produce
    a standardized output. This means stereochemical information is removed and aromatic
    systems are represented with lowercase letters.

    Args:
        smiles: The input SMILES string to sanitize. This may represent a molecule with
            associated salts, counterions, or solvents (indicated by '.' separators in the
            SMILES), or a molecule in non-canonical form. The string should be valid and
            parseable by RDKit's molecular structure parser.

    Returns:
        A sanitized SMILES string representing the main molecular structure in canonical form.
        The output will have salts and solvents removed, with a unique canonical representation
        that excludes stereochemical information and uses aromatic notation for aromatic systems.
        This standardized form is suitable for molecular comparison, database operations, and
        machine learning applications.
    """
    # Remove salts from the SMILES
    smiles = desalt_smiles(smiles)
    # Canonicalize the SMILES
    smiles = canonicalize_smiles(smiles)

    return smiles


def randomize_smiles(smiles: str, isomeric: bool = False, kekule: bool = False) -> tuple[bool, str]:
    """Generate a randomized non-canonical SMILES representation of a molecular structure.

    This function creates an alternative valid SMILES string for the same molecule by randomly
    reordering the atoms and generating a non-canonical SMILES representation. While canonical
    SMILES always produce the same unique string for a given molecule, randomized SMILES provide
    different but chemically equivalent representations. This technique is widely used in machine
    learning applications for molecular property prediction and generative modeling.

    Randomized SMILES are particularly valuable for data augmentation in deep learning models,
    where exposing the model to multiple valid representations of the same molecule during
    training improves generalization and reduces overfitting. By randomly shuffling the atom
    ordering and using non-canonical SMILES generation, this function creates diverse string
    representations that all encode the same molecular structure, effectively expanding the
    training dataset without introducing new chemical entities.

    The function randomly shuffles the atom indices of the molecule and renumbers the atoms
    accordingly before converting back to SMILES. The non-canonical SMILES generation ensures
    that different atom orderings produce different string representations. The function also
    returns a boolean flag indicating whether the randomization actually changed the SMILES
    string, which can be useful for validation or iteration control.

    Args:
        smiles: The input SMILES string to randomize. This should be a valid SMILES string
            that RDKit can parse into a molecular object. The molecule will be converted to
            an internal representation, its atoms will be randomly reordered, and a new
            SMILES string will be generated from this reordered structure.
        isomeric: A flag controlling whether stereochemical information should be preserved
            in the randomized SMILES output. When set to False (default), all stereochemistry
            information (R/S chirality, E/Z double bond configuration) is removed from the
            output. When set to True, the randomized SMILES will include stereochemical
            descriptors (@, @@, /, \) where applicable, maintaining the spatial configuration
            of the original molecule. Defaults to False.
        kekule: A flag controlling the representation of aromatic systems in the output.
            When set to False (default), aromatic atoms are represented with lowercase letters
            (e.g., 'c' for aromatic carbon), providing standard aromatic notation. When set
            to True, aromatic rings are represented using alternating single and double bonds
            (Kekule structure) with explicit bond types, which provides an alternative
            representation without aromatic notation. Defaults to False.

    Returns:
        A tuple containing two elements:
        - A boolean flag indicating whether the randomization successfully produced a different
          SMILES string from the input. This will be True if the randomized SMILES differs from
          the original input, and False if they are identical (which can occasionally happen
          due to random chance or molecular symmetry).
        - The randomized SMILES string representing the same molecular structure as the input
          but with atoms in a different order and non-canonical notation. This string is
          chemically equivalent to the input but provides an alternative textual representation
          suitable for data augmentation or ensemble methods.
    """
    # Convert the SMILES to a molecule
    m = Chem.MolFromSmiles(smiles)
    # Randomize the atom order
    atom_ind_list = list(range(m.GetNumAtoms()))
    np.random.shuffle(atom_ind_list)
    nm = Chem.RenumberAtoms(m, atom_ind_list)
    # Convert back to SMILES
    smiles_rand = Chem.MolToSmiles(
        nm, canonical=False, isomericSmiles=isomeric, kekuleSmiles=kekule
    )
    # Check if the SMILES is different
    rand_flag = smiles != smiles_rand

    return rand_flag, smiles_rand
