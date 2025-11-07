Contributing
============

We welcome contributions to MolNav! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/molnav.git
      cd molnav

3. Install development dependencies:

   .. code-block:: bash

      uv sync --all-extras

4. Create a new branch for your feature:

   .. code-block:: bash

      git checkout -b feature-name

Guidelines
----------

Code Style
~~~~~~~~~~

* Follow PEP 8 style guide
* Use type hints where appropriate
* Write clear, descriptive docstrings

Testing
~~~~~~~

* Write tests for new features
* Ensure all tests pass before submitting a pull request
* Aim for high test coverage

Documentation
~~~~~~~~~~~~~

* Update documentation for new features
* Include docstrings for all public functions and classes
* Add examples when appropriate

Submitting Changes
------------------

1. Commit your changes:

   .. code-block:: bash

      git add .
      git commit -m "Description of changes"

2. Push to your fork:

   .. code-block:: bash

      git push origin feature-name

3. Create a pull request on GitHub

Code of Conduct
---------------

Please be respectful and constructive in all interactions.

Questions?
----------

If you have questions about contributing, please open an issue on GitHub.
