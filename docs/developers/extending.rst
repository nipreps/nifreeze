.. include:: ../links.rst

Extending *NiFreeze*
====================

To extend *NiFreeze*'s functionality,

Creating New Iterators
----------------------

To extend *NiFreeze*'s functionality, such as implementing a custom iterator,
follow these general steps:

1. **Create a new function**: Define a new iterator function implementing the
   desired logic.

   .. code-block:: python

    from typing import Iterator

    def my_iterator(**kwargs) -> Iterator[int]:
        # Size is the number of volumes in the dataset
        size = get_size(**kwargs)

        # Implement custom iteration logic
        index_order = establish_order(size)

        return (x for x in index_order)


1. **Integrate the function**: Incorporate the newly created iterator into the
   *NiFreeze* workflow by updating the configuration or pipeline definitions to
   utilize your new implementation.

   .. code-block:: python

    list(my_iterator(10))


1. **Test the Implementation**: Ensure that your custom class functions as intended by
   running unit tests and validating the outputs against expected results.

Creating New Models
-------------------

To implement a new model:

1. Subclass the :class:`api/nifreeze.model.base.BaseModel` class.
1. Implement the ``fit(data)`` method, which returns a 3D volume.
1. Register the model in your pipeline or CLI wrapper.

Example: Extrapolation model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from nifreeze.data.base import BaseDataset
    from nifreeze.model.base import BaseModel

    class ExtrapolationModel(BaseModel):
        """Model a volume extrapolating across training volumes."""

        __slots__ = {
            "_param1": "Parameter 1",
            "_param2": "Parameter 2",
        }

        def __init__(self, dataset: BaseDataset, param1: float, param2: float, **kwargs):

            super().__init__(dataset, **kwargs)

            self._param1 = param1
            self._param2 = param2

        def _fit(self, index, n_jobs=None, **kwargs):
            # Fit the data
            pass

        def fit_predict(self, index: int, **kwargs):
            # Compute prediction
            return pred

so ``fit_predict`` is suitable for case where the model is fit only once per iteration, and ``predict`` is for cases where
the model is fit across multiple volumes in each iteration.

Usage
^^^^^

.. code-block:: python

    model = ExtrapolationModel()
    result = model.fit(data, index=3)
