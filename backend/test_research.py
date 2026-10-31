"""
Backend Research Literature Test Suite (Simplified Google Scholar Target)

Tests FastAPI research endpoint and Google Scholar link generation:
- GET /api/v1/research/{condition}

Verifies:
- Dynamic Google Scholar URL generation for predicted conditions
- Proper URL encoding of condition names
- Absence of legacy PubMed/Europe PMC status flags
- Clean disclaimer presence
"""

import pytest
from urllib.parse import quote
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

TEST_CONDITIONS = [
    "Melanocytic nevi",
    "Melanoma",
    "Basal cell carcinoma",
    "Actinic keratoses",
    "Benign keratosis-like lesions",
]


@pytest.mark.parametrize("condition", TEST_CONDITIONS)
def test_research_google_scholar_link_generation(condition):
    """Verify endpoint generates correct Google Scholar URL for each predicted condition."""
    encoded_condition = quote(condition)
    response = client.get(f"/api/v1/research/{encoded_condition}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["condition"] == condition
    assert "scholar_url" in data
    assert "scholar.google.com" in data["scholar_url"]
    assert quote(f'"{condition}"') in data["scholar_url"] or quote(condition) in data["scholar_url"]
    assert "disclaimer" in data
    assert "not a medical diagnostic tool" in data["disclaimer"].lower() or "educational" in data["disclaimer"].lower()


def test_research_no_obsolete_pubmed_keys():
    """Verify legacy PubMed/Europe PMC keys are absent from API response."""
    response = client.get("/api/v1/research/Melanoma")
    assert response.status_code == 200
    data = response.json()
    assert "pubmed" not in data
    assert "europe_pmc" not in data
    assert "is_live" not in data
    assert "total_found" not in data
    assert "search_strategy_used" not in data
