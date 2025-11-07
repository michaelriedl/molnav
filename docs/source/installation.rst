Installation
============

Requirements
------------

MolNav requires Python 3.11 or later.

From Source
-----------

To install MolNav from source:

.. code-block:: bash

   git clone https://github.com/michaelriedl/molnav.git
   cd molnav
   uv sync --all-extras

Using pip
---------

.. note::
   MolNav is not yet available on PyPI. Please install from source for now.

Development Installation
------------------------

If you want to contribute to MolNav or modify the code, install in development mode:

.. code-block:: bash

   git clone https://github.com/michaelriedl/molnav.git
   cd molnav
   uv sync --all-extras

This will install all dependencies including development and documentation tools.

Verifying Installation
----------------------

To verify that MolNav is installed correctly, run:

.. code-block:: python

   import molnav
   print(molnav.__version__)
