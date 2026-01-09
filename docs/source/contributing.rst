Contributing
============

We welcome contributions to PlotSmith! Please see the `CONTRIBUTING.md <https://github.com/kylejones200/plotsmith/blob/main/CONTRIBUTING.md>`_ file in the repository for guidelines.

Development Setup
-----------------

1. Fork the repository
2. Clone your fork
3. Install in development mode:

.. code-block:: bash

   pip install -e ".[dev]"

4. Run tests:

.. code-block:: bash

   pytest

5. Run linting:

.. code-block:: bash

   ruff check plotsmith/ tests/

Architecture Guidelines
------------------------

When contributing, please respect the 4-layer architecture:

- **Layer 1** (objects): No matplotlib imports
- **Layer 2** (primitives): Only matplotlib, accepts only Layer 1 objects
- **Layer 3** (tasks): No matplotlib, can use pandas/numpy
- **Layer 4** (workflows): Can use matplotlib, orchestrates tasks and primitives

Code Style
----------

- Use Google-style docstrings
- Include type hints for all functions
- Follow PEP 8 (enforced by ruff)
- Run ``ruff check`` and ``ruff format`` before committing

Testing
-------

- Write tests for new features
- Ensure all tests pass: ``pytest``
- Aim for high test coverage

