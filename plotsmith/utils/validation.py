"""Validation utilities for PlotSmith."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from plotsmith.exceptions import ValidationError

# Type aliases for array-like data
ArrayLike = np.ndarray | pd.Series | list[float] | Sequence[float]
DataContainer = np.ndarray | pd.Series | pd.DataFrame | list | Sequence


def validate_dataframe_columns(
    df: pd.DataFrame, required_columns: list[str], df_name: str = "DataFrame"
) -> None:
    """Validate that a DataFrame contains required columns.

    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
        df_name: Name of the DataFrame for error messages.

    Raises:
        ValidationError: If any required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        available = ", ".join(df.columns.tolist())
        raise ValidationError(
            f"{df_name} missing required columns: {', '.join(missing)}. "
            f"Available columns: {available}",
            context={
                "missing_columns": missing,
                "available_columns": df.columns.tolist(),
                "required_columns": required_columns,
            },
        )


def validate_array_lengths(
    *arrays: ArrayLike, names: list[str] | None = None
) -> None:
    """Validate that multiple arrays have the same length.

    Args:
        *arrays: Variable number of arrays to validate. Accepts:
            - numpy.ndarray
            - pandas.Series
            - list[float]
            - Sequence[float]
        names: Optional list of names for each array (for error messages).

    Raises:
        ValidationError: If arrays have mismatched lengths.

    Returns:
        None: Raises ValidationError if validation fails.
    """
    if len(arrays) < 2:
        return

    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        if names and len(names) == len(arrays):
            length_info = ", ".join(
                f"{name}: {length}" for name, length in zip(names, lengths)
            )
            raise ValidationError(
                f"Arrays must have same length. Got: {length_info}",
                context={"lengths": dict(zip(names, lengths))},
            )
        else:
            raise ValidationError(
                f"Arrays must have same length. Got lengths: {lengths}",
                context={"lengths": lengths},
            )


def validate_not_empty(data: DataContainer, data_name: str = "data") -> None:
    """Validate that data is not empty.

    Args:
        data: Data to validate. Accepts:
            - numpy.ndarray
            - pandas.Series
            - pandas.DataFrame
            - list
            - Any sequence with __len__ method
        data_name: Name of the data for error messages.

    Raises:
        ValidationError: If data is empty.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValidationError(
                f"{data_name} is empty (DataFrame has no rows)",
                context={"data_type": "DataFrame", "shape": data.shape},
            )
    elif isinstance(data, pd.Series):
        if len(data) == 0:
            raise ValidationError(
                f"{data_name} is empty (Series has no elements)",
                context={"data_type": "Series", "length": len(data)},
            )
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValidationError(
                f"{data_name} is empty (array/list has no elements)",
                context={"data_type": type(data).__name__, "length": len(data)},
            )
    else:
        try:
            if len(data) == 0:
                raise ValidationError(
                    f"{data_name} is empty",
                    context={"data_type": type(data).__name__, "length": len(data)},
                )
        except TypeError:
            # Data doesn't support len(), assume it's valid
            pass


def validate_2d_array(
    data: np.ndarray | pd.DataFrame | Sequence[Sequence[float]],
    data_name: str = "data",
) -> None:
    """Validate that data is a 2D array.

    Args:
        data: Data to validate. Accepts:
            - numpy.ndarray (2D)
            - pandas.DataFrame
            - Sequence[Sequence[float]] (nested sequences)
        data_name: Name of the data for error messages.

    Raises:
        ValidationError: If data is not 2D.

    Returns:
        None: Raises ValidationError if validation fails.
    """
    if isinstance(data, pd.DataFrame):
        # DataFrame is always 2D, but check if it's empty
        if data.empty:
            raise ValidationError(
                f"{data_name} is empty (DataFrame has no rows or columns)",
                context={"data_type": "DataFrame", "shape": data.shape},
            )
    else:
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValidationError(
                f"{data_name} must be 2D, got {arr.ndim}D",
                context={
                    "data_type": type(data).__name__,
                    "ndim": arr.ndim,
                    "shape": arr.shape,
                },
            )


def suggest_column_name(
    requested: str, available: list[str], threshold: int = 2
) -> str | None:
    """Suggest a similar column name if the requested one doesn't exist.

    Uses simple string similarity (Levenshtein-like) to find close matches.

    Args:
        requested: The requested column name.
        available: List of available column names.
        threshold: Maximum edit distance for suggestions.

    Returns:
        Suggested column name if found, None otherwise.
    """
    if not available:
        return None

    requested_lower = requested.lower()

    # Exact case-insensitive match
    for col in available:
        if col.lower() == requested_lower:
            return col

    # Find close matches (simple edit distance)
    def simple_distance(s1: str, s2: str) -> int:
        """Simple edit distance calculation."""
        s1, s2 = s1.lower(), s2.lower()
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    best_match = None
    best_distance = float("inf")

    for col in available:
        distance = simple_distance(requested, col)
        if distance < best_distance and distance <= threshold:
            best_distance = distance
            best_match = col

    return best_match

