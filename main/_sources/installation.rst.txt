.. include:: links.rst

Installation
============
Make sure all of *nifreeze*' `External Dependencies`_ are installed.

On a functional Python 3.13 (or above) environment with ``pip`` installed,
*nifreeze* can be installed using the usual ``pip install`` command::

    python -m pip install nifreeze

Check your installation with the following command line::

    python -c "from nifreeze import __version__; print(__version__)"

Supported versions
------------------
*nifreeze* follows `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`__,
the minimum supported dependencies of the Scientific Python ecosystem.
A Python version is supported for three years after its release, and a version of a
dependency for two years after its release.
Python 3.13 and 3.14 are supported, and every pull request is tested on both.
Python 3.15 is tested weekly, and will be supported once all *nifreeze*'s dependencies
publish packages for it.
Each quarter, an automated pull request raises the minimum versions of the dependencies,
and a test job checks that *nifreeze* still works with them.


External Dependencies
---------------------
*nifreeze* requires ANTs_, which is leveraged through the Nipype_ Python
interface for registration purposes. There are
`several ways to install ANTs <https://github.com/ANTsX/ANTs?tab=readme-ov-file#installation>`__.
Notably, the path to the installed binaries needs to be added to the ``PATH``::

   export PATH=/path/to/ants/bin:$PATH
