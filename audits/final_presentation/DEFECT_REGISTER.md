# Comprehensive Defect Register

## 1. Prioritized Matrix & Remediation History

This register tracks all discovered defects across codebase inspection, lint checks, runtime evaluations, and academic presentation standards.

| Defect ID | Severity | Affected Component | Root Cause | Remediation / Fix | Verification | Final Status |
| :--- | :---: | :--- | :--- | :--- | :--- | :---: |
| **DEF-01** | **Low** (Cosmetic) | `frontend/src/components/Disclaimer.jsx` & `InfoPage.jsx` | Stray prompt text in line 1 caused JSX parse errors in linter. | Stripped prompt comments from both files. | `npx oxlint` reduced errors to 0. | **FIXED** |
| **DEF-02** | **Low** (Warning) | `frontend/src/components/ExplainabilityPanel.jsx` | Tooltip coordinates accessed `containerRef.current.getBoundingClientRect()` inside JSX inline styles during render. | Stored relative coordinates within mouse handler; removed ref access during render. | Clean render without React Compiler optimization warnings. | **FIXED** |
| **DEF-03** | **Medium** (Operational) | `backend/.env` vs Live Process | Updating `.env` does not alter an already running server daemon until process reload. | Documented operator startup/reload command in demo runbook. | Preserved running daemon per Rule 4; isolated test client validated 5/5. | **DOCUMENTED** |
| **DEF-04** | **Medium** (Scientific) | Dataset manifests | Lack of patient/lesion grouping IDs prevents ruling out data leakage across splits. | Explicit disclosure across disclaimers and academic runbook. | Mandatory disclosure in faculty presentation script. | **EXPLICITLY DISCLOSED** |
| **DEF-05** | **Medium** (Scientific) | Historical test split | Historical test split was reused across prior development phases. | Preserved formal rating `SIGNOFF_WITH_LIMITATIONS`. | Disclosed in academic documentation and presentation slides. | **EXPLICITLY DISCLOSED** |
| **DEF-06** | **Low** (Modeling) | Closed 14-class taxonomy | Unsupported conditions map to nearest in-distribution class. | Entropy-based uncertainty indicators and disclaimers alert users. | Validated entropy scoring in uncertainty module. | **MITIGATED BY DISCLOSURE** |

