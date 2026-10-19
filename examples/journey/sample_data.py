"""Small synthetic histories, never a production target-generation recipe."""

import calendar
import random
from datetime import date


def make_rows(n_accounts=120):
    """Return base and spine records with three disjoint chronological cohorts."""
    rng = random.Random(42)
    base, spine = [], []
    for i in range(n_accounts):
        month = 4 + min(2, 3 * i // n_accounts)
        ref = date(2026, month, calendar.monthrange(2026, month)[1])
        account = f"A{i:04}"
        target = i % 2
        segment = "small" if (i // 2) % 2 else "large"
        spine.append((account, ref, segment, target))
        for offset in (3, 2, 1):
            observed_month = month - offset
            observed = date(
                2026, observed_month, calendar.monthrange(2026, observed_month)[1]
            )
            revenue = 100 + 20 * target + rng.gauss(0, 25)
            balance = 300 + 30 * target + rng.gauss(0, 50)
            if i % 19:  # Some entities have no source history.
                base.append(
                    (
                        account,
                        observed,
                        revenue,
                        None if i % 11 == 0 and offset == 1 else balance,
                    )
                )
    return base, spine
