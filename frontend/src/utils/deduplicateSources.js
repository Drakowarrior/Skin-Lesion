/**
 * Deduplicate biomedical research sources using reliable identifiers.
 *
 * Priority:
 * 1. PMID (PubMed Unique Identifier)
 * 2. DOI (Digital Object Identifier)
 * 3. Canonical URL
 * 4. Normalized title (fallback only when length > 5)
 *
 * Preserves the first instance and does not merge items with missing identifiers.
 *
 * @param {Array<Object>} sources
 * @returns {Array<Object>} deduplicated sources
 */
export function deduplicateSources(sources) {
  if (!Array.isArray(sources) || sources.length === 0) return [];
  const seen = new Set();
  const result = [];

  for (const src of sources) {
    if (!src || typeof src !== "object") continue;

    let key = null;

    if (src.pmid) {
      key = `pmid:${String(src.pmid).trim().toLowerCase()}`;
    } else if (src.doi) {
      key = `doi:${String(src.doi).trim().toLowerCase()}`;
    } else if (src.url) {
      try {
        const parsed = new URL(src.url);
        key = `url:${parsed.hostname.toLowerCase()}${parsed.pathname.replace(/\/$/, "").toLowerCase()}`;
      } catch {
        key = `url:${String(src.url).trim().toLowerCase()}`;
      }
    } else if (src.title) {
      const normalizedTitle = String(src.title)
        .toLowerCase()
        .replace(/[^a-z0-9]/g, "");
      if (normalizedTitle.length > 5) {
        key = `title:${normalizedTitle}`;
      }
    }

    if (!key) {
      // Missing all identifiers: do not merge with unrelated articles
      result.push(src);
      continue;
    }

    if (!seen.has(key)) {
      seen.add(key);
      result.push(src);
    }
  }

  return result;
}
