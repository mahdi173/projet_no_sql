import uuid
from datetime import datetime

def normalize_field(value, as_type=None):
    if value is None:
        return None

    value = value.strip()

    if value == "" or value.upper() == "NA" or value == ".":
        return None

    if as_type is not None:
        try:
            return as_type(value)
        except:
            return None

    return value

def parse_penguin_row(row):
    """Convert a CSV row into a normalized penguin dict."""
    return {
        "penguin_id": str(uuid.uuid4()),       # Always generate new ID
        "island": row["island"],
        "species": row["species"],
        "sex": normalize_field(row.get("sex")),

        "features": {
            "bill_length_mm": normalize_field(row.get("bill_length_mm"), float),
            "bill_depth_mm": normalize_field(row.get("bill_depth_mm"), float),
            "flipper_length_mm": normalize_field(row.get("flipper_length_mm"), int),
            "body_mass_g": normalize_field(row.get("body_mass_g"), int),
        },
    }