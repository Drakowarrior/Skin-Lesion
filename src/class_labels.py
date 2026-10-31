"""
Canonical Class Label Contract & Medical Terminology Mapping

Defines the single source of truth mapping between model logit/softmax indices,
human-readable skin condition labels, and medical terminology/literature search concepts.
Derived strictly from src/resnet_data.py.
"""

from typing import Dict, List, Tuple, Any

# Canonical 14-class list in exact model index order
CLASS_NAMES: List[str] = [
    "Actinic keratoses",             # 0
    "Basal cell carcinoma",          # 1
    "Benign keratosis-like lesions", # 2
    "Chickenpox",                    # 3
    "Cowpox",                        # 4
    "Dermatofibroma",                # 5
    "HFMD",                          # 6
    "Healthy",                       # 7
    "Measles",                       # 8
    "Melanocytic nevi",              # 9
    "Melanoma",                      # 10
    "Monkeypox",                     # 11
    "Squamous cell carcinoma",       # 12
    "Vascular lesions"               # 13
]

INDEX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

NUM_CLASSES: int = len(CLASS_NAMES)

# Comprehensive Medical Terminology & Progressive Literature Search Mapping
MEDICAL_SEARCH_MAPPING: Dict[str, Dict[str, Any]] = {
    "Actinic keratoses": {
        "medical_concept": "Actinic keratosis",
        "medical_terms": ["actinic keratosis", "solar keratosis", "senile keratosis", "precancerous skin lesion"],
        "search_terms": [
            '"actinic keratosis"',
            '"solar keratosis"',
            '"actinic keratosis" AND dermoscopy',
            '"actinic keratosis" AND diagnosis',
            '"precancerous skin lesion" AND dermoscopy',
        ],
    },
    "Basal cell carcinoma": {
        "medical_concept": "Basal cell carcinoma",
        "medical_terms": ["basal cell carcinoma", "BCC", "rodent ulcer", "basal cell epithelioma"],
        "search_terms": [
            '"basal cell carcinoma"',
            '"BCC" AND dermoscopy',
            '"basal cell carcinoma" AND dermoscopy',
            '"basal cell carcinoma" AND diagnosis',
            '"basal cell carcinoma" AND dermatology',
        ],
    },
    "Benign keratosis-like lesions": {
        "medical_concept": "Seborrheic keratosis",
        "medical_terms": ["seborrheic keratosis", "seborrhoeic keratosis", "benign keratosis", "solar lentigo", "lichen planus-like keratosis"],
        "search_terms": [
            '"seborrheic keratosis"',
            '"seborrhoeic keratosis"',
            '"seborrheic keratosis" AND dermoscopy',
            '"seborrheic keratosis" AND diagnosis',
            '"benign keratosis" AND dermoscopy',
            '"benign skin lesion" AND dermoscopy',
        ],
    },
    "Chickenpox": {
        "medical_concept": "Varicella (Chickenpox)",
        "medical_terms": ["varicella", "chickenpox", "varicella zoster virus", "vesicular rash"],
        "search_terms": [
            '"chickenpox"',
            '"varicella zoster"',
            '"varicella" AND diagnosis',
            '"varicella rash" AND dermatology',
            '"varicella zoster virus" AND clinical presentation',
        ],
    },
    "Cowpox": {
        "medical_concept": "Cowpox virus infection",
        "medical_terms": ["cowpox", "cowpox virus", "orthopoxvirus", "bovine vaccinia"],
        "search_terms": [
            '"cowpox"',
            '"cowpox virus"',
            '"orthopoxvirus" AND skin lesions',
            '"cowpox" AND clinical diagnosis',
        ],
    },
    "Dermatofibroma": {
        "medical_concept": "Dermatofibroma",
        "medical_terms": ["dermatofibroma", "benign fibrous histiocytoma", "cutaneous fibrous histiocytoma"],
        "search_terms": [
            '"dermatofibroma"',
            '"benign fibrous histiocytoma"',
            '"dermatofibroma" AND dermoscopy',
            '"dermatofibroma" AND diagnosis',
        ],
    },
    "HFMD": {
        "medical_concept": "Hand, foot and mouth disease",
        "medical_terms": ["hand foot and mouth disease", "HFMD", "coxsackievirus", "enterovirus A71"],
        "search_terms": [
            '"hand foot and mouth disease"',
            '"HFMD" AND skin rash',
            '"coxsackievirus" AND skin manifestation',
            '"hand foot and mouth" AND diagnosis',
        ],
    },
    "Healthy": {
        "medical_concept": "Normal skin histology and dermoscopy",
        "medical_terms": ["normal skin", "healthy skin", "skin barrier", "normal skin dermoscopy"],
        "search_terms": [
            '"normal skin" AND dermoscopy',
            '"healthy skin" AND dermatology',
            '"normal skin" AND morphology',
        ],
    },
    "Measles": {
        "medical_concept": "Measles (Rubeola)",
        "medical_terms": ["measles", "rubeola", "morbillivirus", "measles rash"],
        "search_terms": [
            '"measles"',
            '"rubeola" AND rash',
            '"measles" AND exanthem',
            '"measles" AND clinical diagnosis',
        ],
    },
    "Melanocytic nevi": {
        "medical_concept": "Melanocytic nevus",
        "medical_terms": ["melanocytic nevus", "melanocytic nevi", "benign nevus", "pigmented nevus"],
        "search_terms": [
            '"melanocytic nevus"',
            '"melanocytic nevi"',
            '"melanocytic nevus" AND dermoscopy',
            '"benign nevus" AND dermoscopy',
            '"melanocytic nevi" AND diagnosis',
        ],
    },
    "Melanoma": {
        "medical_concept": "Cutaneous melanoma",
        "medical_terms": ["cutaneous melanoma", "malignant melanoma", "melanoma dermoscopy", "skin melanoma"],
        "search_terms": [
            '"cutaneous melanoma"',
            '"malignant melanoma"',
            '"melanoma" AND dermoscopy',
            '"melanoma" AND diagnosis',
            '"melanoma" AND early detection',
        ],
    },
    "Monkeypox": {
        "medical_concept": "Mpox (Monkeypox)",
        "medical_terms": ["mpox", "monkeypox", "monkeypox virus", "mpox lesion"],
        "search_terms": [
            '"monkeypox"',
            '"mpox"',
            '"mpox" AND skin lesions',
            '"monkeypox virus" AND dermatology',
            '"mpox" AND clinical diagnosis',
        ],
    },
    "Squamous cell carcinoma": {
        "medical_concept": "Squamous cell carcinoma",
        "medical_terms": ["squamous cell carcinoma", "SCC", "cutaneous squamous cell carcinoma", "cSCC"],
        "search_terms": [
            '"squamous cell carcinoma"',
            '"cutaneous squamous cell carcinoma"',
            '"squamous cell carcinoma" AND dermoscopy',
            '"SCC" AND skin diagnosis',
        ],
    },
    "Vascular lesions": {
        "medical_concept": "Vascular skin lesion",
        "medical_terms": ["vascular lesion", "hemangioma", "cherry angioma", "pyogenic granuloma", "angioma"],
        "search_terms": [
            '"vascular lesion" AND dermoscopy',
            '"hemangioma" AND dermoscopy',
            '"cherry angioma"',
            '"vascular skin lesion" AND diagnosis',
        ],
    },
}


def get_class_labels() -> Tuple[List[str], Dict[int, str], Dict[str, int]]:
    """Return (CLASS_NAMES, INDEX_TO_CLASS, CLASS_TO_INDEX)."""
    return CLASS_NAMES, INDEX_TO_CLASS, CLASS_TO_INDEX


def get_medical_search_config(class_name: str) -> Dict[str, Any]:
    """Return medical search configuration for a given canonical model class name."""
    for name, config in MEDICAL_SEARCH_MAPPING.items():
        if name.lower() == class_name.strip().lower():
            return config
    clean = class_name.strip()
    return {
        "medical_concept": clean,
        "medical_terms": [clean],
        "search_terms": [f'"{clean}"', f'"{clean}" AND dermoscopy'],
    }
