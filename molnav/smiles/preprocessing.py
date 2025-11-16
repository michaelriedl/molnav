from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


def canonicalize_smiles(smiles: str, isomeric: bool = False, kekule: bool = True) -> str:
    """Canonicalize a SMILES string using RDKit's canonicalization algorithm.

    This function converts a SMILES (Simplified Molecular Input Line Entry System) string
    into its canonical form, ensuring that the same molecule always receives the same SMILES
    representation regardless of the input format. Canonicalization is essential for molecular
    comparison, database searching, and deduplication of chemical structures.

    The function provides options to control the representation of stereochemistry and
    aromaticity in the output SMILES string. By default, stereochemical information is
    removed and Kekule form (with explicit double bonds instead of aromatic notation) is used.

    Args:
        smiles: The input SMILES string representing the molecular structure to be canonicalized.
            This should be a valid SMILES string that RDKit can parse into a molecular object.
        isomeric: A flag controlling whether stereochemical information should be preserved
            in the canonical SMILES output. When set to False (default), all stereochemistry
            information (R/S chirality, E/Z double bond configuration) is removed from the
            output. When set to True, the canonical SMILES will include stereochemical
            descriptors (@, @@, /, \) where applicable. Defaults to False.
        kekule: A flag controlling the representation of aromatic systems in the output.
            When set to True (default), aromatic rings are represented using alternating
            single and double bonds (Kekule structure) rather than lowercase aromatic atom
            symbols. This provides a more explicit representation of the bonding. When set
            to False, aromatic atoms are represented with lowercase letters (e.g., 'c' for
            aromatic carbon). Defaults to True.

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

    Note that the function returns a non-canonical, non-isomeric Kekule SMILES string. If you
    need a canonical representation, consider using `canonicalize_smiles()` on the output.

    Args:
        smiles: The input SMILES string that may contain salts, counterions, or solvent molecules
            in addition to the main molecular structure. This should be a valid SMILES string that
            RDKit can parse. Multiple molecular fragments in the SMILES (indicated by '.' separator)
            will be processed, with salt/solvent components removed according to RDKit's default
            salt definitions.

    Returns:
        A SMILES string representing the desalted molecular structure with salts and solvents
        removed. The output is in Kekule form (explicit single/double bonds) without stereochemical
        information. If the input consists entirely of salt/solvent molecules, the function will
        preserve at least one fragment to avoid returning an empty result (due to the
        `dontRemoveEverything=True` parameter).
    """
    # Initialize the salt/solvent remover
    remover = SaltRemover()
    # Convert the SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    # Remove the salts/solvents
    mol = remover.StripMol(mol, dontRemoveEverything=True)
    # Convert back to SMILES
    smiles = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False, kekuleSmiles=True)

    return smiles
