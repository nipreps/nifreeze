
.. include:: links.rst

Installation
============
Make sure all of *nifreeze*' `External Dependencies`_ are installed.

On a functional Python 3.10 (or above) environment with ``pip`` installed,
*nifreeze* can be installed using the habitual command ::

    $ python -m pip install nifreeze

Check your installation with the following command line ::

    $ python -c "from nifreeze import __version__; print(__version__)"


External Dependencies
---------------------
*nifreeze* requires ANTs_, which is leveraged through the Nipype_ Python
interface for registration purposes. An installation guide can be found
at [this location]. Note that its install location needs to be added to the ``PATH`` ::

   $ export PATH=$ANTSPATH:$PATH
