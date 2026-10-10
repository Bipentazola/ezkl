#!/usr/bin/env python3

import math
import sys
from dataclasses import dataclass

MIN_LOGROWS, RESERVED_BLINDING_ROWS = 6, 8
EZKL_MAX_PUBLIC_SRS = 26
MAX_PUBLIC_SRS = 25

COEF = {
    "setup_wall_s": (-11.7967, 0.9650),
    "prove_wall_s": (-12.2642, 0.9840),
    "setup_rss_mb": (-4.8444, 0.9934),
    "prove_rss_mb": (-3.7462, 0.9626),
}

FIT_LOGROWS_MIN, FIT_LOGROWS_MAX = 16, 21
OOM_ANCHOR = dict(
    logrows=23, observed_partial_setup_rss_mb=158_427.8, status="OOM_OR_KILLED"
)


def logrows_estimate(num_rows: int) -> int:
    return min(
        max(MIN_LOGROWS, math.ceil(math.log2(num_rows + RESERVED_BLINDING_ROWS))),
        MAX_PUBLIC_SRS,
    )


def _cost(field: str, logrows: int) -> float:
    a, b = COEF[field]
    return 2 ** (a + b * logrows)


def _confidence(logrows: int) -> str:
    if FIT_LOGROWS_MIN <= logrows <= FIT_LOGROWS_MAX:
        return "interpolated (within fitted data range 16-21) -- mean_abs residual 1-14% depending on field, see docs/recon/05_cost_model.md"
    if logrows == FIT_LOGROWS_MAX + 1:
        return (
            "extrapolated +1 logrows beyond fit domain -- power-law form is well-supported "
            "(R^2>=0.98) but only partly verified at this logrows: `setup` has since been "
            "measured twice at 22 (129.2 GiB for seq8_d384_l1, 132.9 GiB for the optimized "
            "seq16_d384_l1 prefill circuit, both within +3.1% of this model), while every `prove` at 22 was "
            "OOM-killed, so the prove numbers here remain unverified "
            "(see docs/recon/05_cost_model.md Section 2)"
        )
    if logrows >= OOM_ANCHOR["logrows"]:
        return (
            f"extrapolated {logrows - FIT_LOGROWS_MAX} logrows beyond fit domain -- "
            f"at logrows={OOM_ANCHOR['logrows']} the real machine OOM'd during setup "
            f"(partial peak {OOM_ANCHOR['observed_partial_setup_rss_mb']:,.0f}MB before kill, "
            f"consistent with but necessarily lower than this model's predicted true peak) -- "
            f"treat this estimate as order-of-magnitude only, not a resource-planning number"
        )
    return "extrapolated below the fit domain (logrows<16) -- unverified, but cost only decreases, so risk is low"


@dataclass
class CostEstimate:
    num_rows: int
    logrows: int
    setup_wall_s: float
    prove_wall_s: float
    setup_rss_mb: float
    prove_rss_mb: float
    confidence: str


def estimate(num_rows: int) -> CostEstimate:
    logrows = logrows_estimate(num_rows)
    needed = required_k(num_rows)
    confidence = _confidence(logrows)
    if needed > MAX_PUBLIC_SRS:
        confidence = (
            f"NOT REPRESENTABLE -- needs logrows {needed}, past the representable "
            f"max k = {MAX_PUBLIC_SRS} ({2 ** MAX_PUBLIC_SRS:,} rows) for this circuit family: "
            "constraint degree 6 needs an extended domain of k+3 bits, and BN254's "
            "2-adicity is 28 (keygen panics at k=26, results/k26_keygen_probe/). "
            "This is a capacity wall, not a memory shortage. The costs below are "
            f"for a clamped 2^{MAX_PUBLIC_SRS} circuit and are a lower bound only. || "
            + confidence
        )
    return CostEstimate(
        num_rows=num_rows,
        logrows=logrows,
        setup_wall_s=_cost("setup_wall_s", logrows),
        prove_wall_s=_cost("prove_wall_s", logrows),
        setup_rss_mb=_cost("setup_rss_mb", logrows),
        prove_rss_mb=_cost("prove_rss_mb", logrows),
        confidence=confidence,
    )


def required_k(num_rows: int) -> int:
    return max(MIN_LOGROWS, (num_rows + RESERVED_BLINDING_ROWS - 1).bit_length())


def within_srs_capacity(num_rows: int) -> bool:
    return required_k(num_rows) <= MAX_PUBLIC_SRS


def feasible_on(num_rows: int, budget_mb: float) -> bool:
    if not within_srs_capacity(num_rows):
        return False
    e = estimate(num_rows)
    return max(e.setup_rss_mb, e.prove_rss_mb) <= budget_mb


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/estimate_cost.py [arguments]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        e = estimate(int(arg))
        print(f"num_rows={e.num_rows:,}  logrows={e.logrows}")
        print(f"  setup:  {e.setup_wall_s:>10.1f}s   {e.setup_rss_mb:>12,.0f}MB")
        print(f"  prove:  {e.prove_wall_s:>10.1f}s   {e.prove_rss_mb:>12,.0f}MB")
        print(f"  confidence: {e.confidence}")
        print()


if __name__ == "__main__":
    main()
