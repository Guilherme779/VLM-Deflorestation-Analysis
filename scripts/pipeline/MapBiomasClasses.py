from dataclasses import dataclass

@dataclass
class MapBiomasClasses:
    IGNORE = 255
    CLASS_NAMES = ["forest", "pasture", "agriculture", "water", "urban"]
    CLASS_LABELS = {
        "forest": "forest",
        "pasture": "pasture",
        "agriculture": "agriculture",
        "water": "water",
        "urban": "urban",
    }