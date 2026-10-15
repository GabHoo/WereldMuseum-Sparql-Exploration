"""Metadata-based internal diversity for a selected exhibition.

Version 1 computes a Shannon entropy for each descriptive metadata field and
returns their unweighted mean. Keeping fields separate avoids mixing unrelated
categories such as object types and production dates into one distribution.
"""
from collections import Counter
from math import log
from typing import Mapping, Sequence


# Fields that describe the objects. High-cardinality/free-text fields such as
# titles and descriptions are omitted so they do not make any selection appear
# diverse merely because every object's wording is unique.
DEFAULT_METADATA_FIELDS = (
    "type",
    "material",
    "place",
    "date",
    "maker",
    "intended_use",
    "acquisition",
    "acquired_from",
)


def calculate_shannon_index(frequencies: Mapping[str, float]) -> float:
    """Return Shannon entropy (in nats) for a mapping of category frequencies."""
    total = sum(count for count in frequencies.values() if count > 0)
    if total == 0:
        return 0.0

    return -sum(
        (count / total) * log(count / total)
        for count in frequencies.values()
        if count > 0
    )


def calculate_selection_diversity(
    selected_objects: Sequence[Mapping],
    fields: Sequence[str] | None = None,
) -> float:
    """Calculate metadata diversity for the selected objects.

    Objects may be raw artifact dictionaries or pool items with their artifact
    metadata nested under ``properties``. For each field, a metadata category
    contributes once per object that contains it, even if repeated in that
    object's value list. Fields with no values are skipped.

    The returned score is the unweighted mean of the per-field Shannon indices
    (natural logarithm, nats). It is 0 when fewer than two distinct categories
    are represented, or when no configured fields have values.
    """
    metadata_fields = fields if fields is not None else DEFAULT_METADATA_FIELDS
    per_field_indices = []

    for field in metadata_fields:
        frequencies: Counter[str] = Counter()
        for selected_object in selected_objects:
            properties = selected_object.get("properties", selected_object)
            value = properties.get(field)
            if isinstance(value, str):
                values = [value]
            elif isinstance(value, (list, tuple, set)):
                values = value
            else:
                continue

            # A single artifact should not gain extra weight from duplicate
            # values in its metadata list.
            categories = {
                category.strip().casefold()
                for category in values
                if isinstance(category, str) and category.strip()
            }
            if categories:
                # Multi-valued metadata is split evenly so every artifact
                # contributes a total weight of one to this field.
                weight = 1.0 / len(categories)
                frequencies.update({category: weight for category in categories})

        if frequencies:
            per_field_indices.append(calculate_shannon_index(frequencies))

    if not per_field_indices:
        return 0.0
    return sum(per_field_indices) / len(per_field_indices)
