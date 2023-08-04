.. image:: ./docs/static/images/logo_and_title.png
    :target: https://github.com/AlanSynn/deepcrunch
    :alt: DeepCrunch
    :align: center

.. |centered text| raw:: html

    <h1 style="text-align: center;">DeepCrunch: A Comprehensive DNN Compressor for General Usage</h1>

|centered text|

|lguplus| |pytorch-extension| |license| |release|
|lint| |test| |codecov| |docs| |pre-commit|
|python| |c++| |cuda| |black| |made-with-love|

-----

Project Origin & Overview
=========================

This project, `DeepCrunch`, is a product of an internship opportunity with LG U+ conducted by `Alan Synn (alan@alansynn.com)`. `Alan`, while being part of the CDO MLOps team, dedicated skills and effort into crafting this open-source Python library.

The inception of `DeepCrunch` happened with the objective of providing a robust platform to support popular model compression techniques. Unlike many other utilities, `DeepCrunch` is not restricted to one or two frameworks and backends. Instead, it is designed to support multi-frameworks and backends with all mainstream deep learning frameworks - TensorFlow, PyTorch, ONNX Runtime, and MXNet.

Currently, `DeepCrunch` aims to support checkpoint compression for model training. This feature is currently under development and will be released in the near future.

-----

Usage
=====

DeepCrunch is designed to be user-friendly and easy to integrate into your machine learning workflow. To get you started quickly, we provide a set of tutorials and example scripts in the `examples` directory. We also have detailed API documentation and guides on our `Documentation`_.

.. _Documentation: https://AlanSynn.github.io/deepcrunch/en/latest/

Please refer to our `USAGE.rst <./USAGE.rst>`_ for a comprehensive guide on how to use DeepCrunch in your projects.

Installation
============

**Install dependencies**

To utilize the functionalities of DeepCrunch, start by creating a conda environment as follows:

.. code-block:: bash

    conda env create -f environment.yml -p ./env

Build
=====

**Build Python package for development**

.. code-block:: bash

    conda activate ./env
    make build-dev

**Build Python package**

.. code-block:: bash

    conda activate ./env
    make build

**Install Python package**

.. code-block:: bash

    make install

**Appendix: Clean Build**

.. code-block:: bash

    conda activate ./env
    make clean-build

**Appendix: Reinstall with clean build**

.. code-block:: bash

    conda activate ./env
    make reinstall

Code formatting
===============

**Get linting results**

.. code-block:: bash

    conda activate ./env
    make lint

**Automatically format code**

.. code-block:: bash

    conda activate ./env
    make format

-----

Milestones
==========

The `MILESTONES.rst` file outlines the significant achievements and benchmarks we've reached so far in the development of DeepCrunch, and also lays out our roadmap for future development.

Some key milestones include:

- Initial release with support for PyTorch (v0.1.0)
- Checkpoint Compression support for PyTorch (v0.2.0)
- Expanded support to include ONNX Runtime (v1.0.0)

For a detailed overview of the project milestones and to gain insight into our future plans, please refer to the `MILESTONES.rst` document linked below.

See `MILESTONES.rst <./MILESTONES.rst>`_.

Changelog
=========

DeepCrunch is under active development, and we strive to regularly update our users about the changes, improvements, and fixes we introduce in each version of the library. The `CHANGELOG.rst` file serves as a record of all these updates, providing transparency and keeping users informed about what changes to expect in each new version of DeepCrunch.

The changelog contains details such as:

- New features added
- Improvements made to existing features
- Bug fixes
- Any breaking changes

For a detailed list of updates, enhancements, and fixes in each version, please refer to the `CHANGELOG.rst` document linked below.

See `CHANGELOG.rst <./CHANGELOG.rst>`_.

Contributing
============

We welcome and appreciate contributions from the community! Whether it's reporting bugs, proposing new features, improving documentation, or contributing code, your help makes DeepCrunch better.

Please refer to our `CONTRIBUTING.rst <./CONTRIBUTING.rst>`_ to understand our contribution guidelines and the process for submitting pull requests to us.

Support
=======

If you encounter any issues or have questions about DeepCrunch, feel free to open an issue on our GitHub repository. Our team will do their best to assist you.

-----

License
=======

DeepCrunch is a proprietary software owned by LG U+. All rights reserved. Please see `LICENSE <./LICENSE>`_ for more information on the terms of use.

.. Aliases

.. |repo| replace:: https://github.com/AlanSynn/deepcrunch

.. Statics

.. |lguplus| image:: https://img.shields.io/badge/Global_Summer_Internship-D60078.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9Ijk2Ljc2IDAuMTUgNDMuMSAzMS44MSI+PHBhdGggZmlsbD0iI0ZGRkZGRiIgZD0iTTEzMy42NTkgNi40NjdWLjE0NmgtNC42NjR2Ni4zMjFoLTYuMTY1djQuNDU5aDYuMTY1djYuMjQ3aDQuNjY0di02LjI0N2g2LjIwN1Y2LjQ2N2gtNi4yMDdaTTExMy4xNiA2LjQ5NVYyMC45OWMwIDQuMDA2LTEuOTg5IDUuOTc2LTUuMzU0IDUuOTc2cy01LjM1NC0xLjk3LTUuMzU0LTUuOTc2VjYuNDk1aC01LjY4OHYxNC41OGMwIDcuOTEyIDQuMjA5IDEwLjg4MiAxMC44MzggMTAuODgybC4yMDQtLjAwMS4yMDQuMDAxYzYuNjMgMCAxMC44NC0yLjk3MSAxMC44NC0xMC44ODJWNi40OTVoLTUuNjl6Ii8+PC9zdmc+
    :target: https://www.uplus.co.kr/
    :alt: LG U+ Global Summer Internship
.. |pytorch-extension| image:: https://img.shields.io/badge/PyTorch_Extension-%23EE4C2C.svg?logo=pytorch&logoColor=white
    :target: https://www.uplus.co.kr/
    :alt: PyTorch Extension
.. |license| image:: https://img.shields.io/badge/license-proprietary-blue
    :target: |repo|/LICENSE
    :alt: Propritary License by U+
.. |release| image:: https://img.shields.io/github/release/AlanSynn/deepcrunch
    :target: https://github.com/AlanSynn/deepcrunch
    :alt: Release
.. |lint| image:: https://github.com/AlanSynn/deepcrunch/actions/workflows/lint.yml/badge.svg
    :target: https://github.com/AlanSynn/deepcrunch/actions/workflows/lint.yml
    :alt: Linting
.. |test| image:: https://github.com/AlanSynn/deepcrunch/actions/workflows/test.yml/badge.svg
    :target: https://github.com/AlanSynn/deepcrunch/actions/workflows/test.yml
    :alt: Testing
.. |codecov| image:: https://codecov.io/gh/AlanSynn/deepcrunch/branch/main/graph/badge.svg?token=UFSCNCO5AZ
    :target: https://codecov.io/gh/AlanSynn/deepcrunch
    :alt: Code Coverage
.. |docs| image:: https://github.com/AlanSynn/deepcrunch/actions/workflows/docs.yml/badge.svg
    :target: https://github.com/AlanSynn/deepcrunch/actions/workflows/docs.yml
    :alt: Docs
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
    :target: https://github.com/pre-commit/pre-commit
    :alt: Pre-commit enabled
.. |made-with-love| image:: https://img.shields.io/badge/Made_with_♥_by_Alan_Synn-D60078.svg
    :target: https://alansynn.com
    :alt: Made with Love by Alan Synn
.. |c++| image:: https://img.shields.io/badge/c++-%2300599C.svg?logo=c%2B%2B&logoColor=white
    :target: https://www.iso.org/
    :alt: C++
.. |python| image:: https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white
    :target: https://www.python.org/
    :alt: Python
.. |cuda| image:: https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white
    :target: https://developer.nvidia.com/cuda
    :alt: CUDA
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000?logo=black
    :target: https://github.com/psf/black
    :alt: Black code style
