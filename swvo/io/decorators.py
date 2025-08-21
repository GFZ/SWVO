# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from functools import wraps
from datetime import datetime


def add_time_docs(action=None):
    """
    A decorator to add start_time and end_time parameters to the docstring.
    It will add them at the beginning of the Parameters section.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        params = inspect.signature(func).parameters
        original_doc = func.__doc__ or ""

        if "Parameters" in original_doc:
            pre_params = original_doc[: original_doc.find("Parameters")]

            new_params = "Parameters\n"
            new_params += "        ----------\n"

            if "start_time" in params:
                if action:
                    new_params += f"        start_time : datetime\n            Start time for data {action}.\n"
                else:
                    new_params += "        start_time : datetime\n"
            if "end_time" in params:
                if action:
                    new_params += f"        end_time : datetime\n            End time for data {action}.\n"
                else:
                    new_params += "        end_time : datetime\n"

            param_start = original_doc.find("Parameters\n        ----------\n") + len(
                "Parameters\n        ----------\n"
            )
            param_end = (
                original_doc.find("\n\nReturns")
                if "\n\nReturns" in original_doc
                else len(original_doc)
            )
            existing_params = original_doc[param_start:param_end].strip()

            new_params += "        "
            new_params += existing_params

            if "Returns" in original_doc:
                returns_section = original_doc[original_doc.find("\n\nReturns") :]
                new_doc = f"{pre_params}{new_params}{returns_section}"
            else:
                new_doc = f"{pre_params}{new_params}"

        else:
            # No Parameters section, add it before Returns
            pre_params = original_doc[: original_doc.find("Returns")]
            new_doc = f"{pre_params}Parameters\n        ----------\n"

            if "start_time" in params:
                if action:
                    new_doc += f"        start_time : datetime\n            Start time for data {action}.\n"
                else:
                    new_doc += "        start_time : datetime\n"
            if "end_time" in params:
                if action:
                    new_doc += f"        end_time : datetime\n            End time for data {action}.\n"
                else:
                    new_doc += "        end_time : datetime\n"
            new_doc += "\n        "
            new_doc += original_doc[original_doc.find("Returns") :]

        wrapper.__doc__ = new_doc
        wrapper.__annotations__ = func.__annotations__
        return wrapper

    return decorator


def add_attributes_to_class_docstring(cls):
    """
    Automatically adds attributes to the class docstring,
    including inherited attributes for derived classes.
    """
    # Traverse through the class and its bases
    attributes = {}
    for base in cls.__mro__:  # Method Resolution Order
        attributes.update(
            {
                attr: value
                for attr, value in vars(base).items()
                if not attr.startswith("__") and not callable(value)
            }
        )

    if not attributes:
        return cls

    # Update the docstring
    doc = cls.__doc__ or ""
    doc += "\n    Attributes\n    ----------\n"

    for attr, value in attributes.items():
        doc += f"    {attr} : {type(value).__name__} = {value}\n"

    cls.__doc__ = doc

    return cls


def add_methods_to_class_docstring(cls):
    """
    Automatically adds methods to the class docstring,
    including inherited methods for derived classes.
    """
    # Traverse through the class and its bases
    methods = {}
    for base in cls.__mro__:  # Method Resolution Order
        methods.update(
            {
                attr: value
                for attr, value in vars(base).items()
                if not (attr.startswith("__") or attr.startswith("_")) and callable(value)
            }
        )

    if not methods:
        return cls

    # Update the docstring
    doc = cls.__doc__ or ""
    doc += "\n    Methods\n    -------\n"

    for method_name in methods:
        doc += f"    {method_name}\n"

    cls.__doc__ = doc

    return cls
