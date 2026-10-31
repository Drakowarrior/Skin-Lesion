# Research Deduplication & Google Scholar Verification

## 1. Architectural Literature Separation

The application enforces a strict separation between **verified peer-reviewed clinical evidence** and **external exploratory search**:
1. **Patient Guidance / Evidence Section:** Verified PubMed citations with PMIDs directly linked to official NIH/NLM databases.
2. **Dedicated Research Section:** Exactly one external search card for Google Scholar targeting the predicted condition.

---

## 2. Multi-tier Source Deduplication Algorithm

The source deduplication engine in `frontend/src/utils/deduplicateSources.js` was evaluated against duplicate articles, near-duplicate titles, and varied identifier formats.

### Deduplication Hierarchy:
1. **PMID (Primary):** Normalized PubMed ID (e.g. `pmid:34215567`).
2. **DOI (Secondary):** Normalized Digital Object Identifier (e.g. `doi:10.1001/jamadermatol.2021.1234`).
3. **Canonical URL (Tertiary):** Hostname + normalized pathname.
4. **Normalized Title (Fallback):** Lowercase alphanumeric title comparison when title length $> 5$.
5. **Unidentified Preservation:** Articles lacking all identifiers are retained rather than accidentally merged.

### Verification Outcome:
- Evaluated with 10 mixed source records (duplicates by PMID, DOI, URL, Title):
  - Input count: $10$ records.
  - Deduplicated output count: $6$ unique records.
  - Verification Status: **PASS**.

---

## 3. Dedicated Google Scholar Action Card

- **Location:** Strictly in `frontend/src/components/ResearchSection.jsx`.
- **Query Encoding:** `https://scholar.google.com/scholar?q=${encodeURIComponent('"' + predictedClass + '"')}`
- **Redundancy Check:**
  - Scholar links in `ResearchSection.jsx`: **1** (Dedicated Card)
  - Scholar links in `PatientGuidanceSection.jsx`: **0**
  - Scholar links in `TopPredictions.jsx`: **0**
  - Scholar links in `PredictionCard.jsx`: **0**
- **Disclaimers:** The dedicated card explicitly states:
  > *"Opens Google Scholar in an external tab to explore published clinical research and scientific studies. Google Scholar search results are independent external sources and are not automatically verified by this application."*

