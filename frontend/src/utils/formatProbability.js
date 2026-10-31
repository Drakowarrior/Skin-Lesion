/**
 * Centralized Probability Formatter for DermaVision AI (Phase 7E)
 * 
 * Scientifically Honest Formatting Rules:
 * 1. Never round a value below 100% to 100.00%.
 * 2. For probabilities extremely close to 100%, show enough decimal places to reveal actual value (e.g. 99.996%).
 * 3. For small nonzero probabilities, show '< 0.01%' or sufficient precision instead of misleading 0.00%.
 * 4. If the probability is float-exact 1.0, report 100% as model-assigned probability.
 * 5. Strictly preserves raw calibrated floating point numbers without capping or fabrication.
 */

export function formatProbability(p) {
  if (p === null || p === undefined || isNaN(p)) return "N/A";
  const num = typeof p === "number" ? p : parseFloat(p);
  if (isNaN(num)) return "N/A";

  const pct = num * 100;

  // Float-exact 1.0
  if (num >= 1.0) {
    return "100.00%";
  }

  // Near 1.0 (< 100.0% but >= 99.999%)
  if (pct >= 99.999) {
    return `${pct.toFixed(4)}%`;
  }
  if (pct >= 99.99) {
    return `${pct.toFixed(3)}%`;
  }
  if (pct >= 99.9) {
    return `${pct.toFixed(2)}%`;
  }

  // Small non-zero probabilities (prevent misleading 0.00%)
  if (pct > 0 && pct < 0.0001) {
    return "< 0.0001%";
  }
  if (pct > 0 && pct < 0.01) {
    return "< 0.01%";
  }

  // Standard representation (2 decimal places)
  return `${pct.toFixed(2)}%`;
}

export function formatProbabilityDetailed(p) {
  if (p === null || p === undefined || isNaN(p)) return "N/A";
  const num = typeof p === "number" ? p : parseFloat(p);
  if (isNaN(num)) return "N/A";
  const pct = num * 100;
  return `${pct.toFixed(4)}% (p = ${num.toFixed(6)})`;
}
