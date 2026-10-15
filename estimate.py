"""
Estimating turnover from debtors, for the filings where it is not on the
public record.

A small company filing filleted accounts under section 444 does not deliver
a profit and loss account to the registrar, so turnover is genuinely absent.
Working back from debtors is a reasonable way to get an order of magnitude,
provided three things are kept straight.

Use trade debtors, not total debtors
------------------------------------
The balance sheet Debtors line is everything owed to the company. On the
sample filing it breaks down as:

    Trade debtors                       312,365
    Other debtors                        65,500
    Other debtors, due after one year   115,847   a former director loan
                                        -------
    Total                               493,712

Only the first line came from selling something. Estimating off the total
overstates turnover by 58% on this company alone, and the gap is worse
wherever there is a large director loan or intercompany balance.

The multiple is fixed at 5
--------------------------
Five implies 73 debtor days. Most UK SMEs run 45 to 60, which would be 6 to
8 times, and the observed ratios on real filings so far range from about 5.3
to 11.5. Five is therefore likely to understate turnover rather than
overstate it, which is the safer direction to be wrong in for a first pass.

It is a constant in this file rather than an environment variable. Change it
here when there is enough evidence to. The batch summary reports the ratios
actually observed on companies that filed both figures, so the evidence
accumulates on its own.

An estimate is never a filed figure
-----------------------------------
The estimate is returned in its own field with the working shown. It never
overwrites `turnover`. If an estimate ends up in the turnover column of a
CRM, six months later nobody remembers which numbers were filed and which
were inferred, and someone quotes one to a seller.
"""

import os
from typing import Any, Dict, Optional

# Fixed at 5 by decision, not read from the environment. It was briefly a
# setting, and a calibration run on five companies suggested 0.0 because two
# of the three observed ratios came from filings that had been misread. A
# multiplier that can be set to zero from a shell variable is a multiplier
# that will silently zero out a whole batch one day.
#
# To change it, edit this line. See observed_multiple() below and the
# calibration figures the batch summary reports, but treat those as evidence
# to weigh rather than a number to paste in.
MULTIPLE = 5.0

# Below this the estimate is noise: a company with a few thousand pounds of
# debtors at a year end tells you nothing about its annual sales.
MIN_DEBTORS = float(os.getenv("MIN_DEBTORS_FOR_ESTIMATE", "1000"))

# Ratios outside this range mean one of the two figures was misread, not
# that the company collects unusually fast or slow. They are excluded from
# the calibration summary so a broken row cannot drag the median to zero.
PLAUSIBLE_RATIO = (1.0, 40.0)


def estimate_turnover(trade_debtors: Optional[float],
                      total_debtors: Optional[float],
                      multiple: float = None) -> Optional[Dict[str, Any]]:
    """Return an estimate with its working, or None if there is no basis.

    Prefers trade debtors. Falls back to total debtors and says so, because
    an estimate off the total is materially weaker and the caller should be
    able to see which one it got.
    """
    multiple = MULTIPLE if multiple is None else multiple

    if trade_debtors is not None and trade_debtors >= MIN_DEBTORS:
        base, basis = trade_debtors, "trade debtors"
        confidence = "medium"
    elif total_debtors is not None and total_debtors >= MIN_DEBTORS:
        base, basis = total_debtors, "total debtors"
        confidence = "low"
    else:
        return None

    value = round(base * multiple)
    return {
        "value": float(value),
        "basis": basis,
        "basis_value": base,
        "multiple": multiple,
        "confidence": confidence,
        "working": f"{basis} {base:,.0f} x {multiple:g} = {value:,}",
        "implied_debtor_days": round(365 / multiple),
    }


def observed_multiple(turnover: Optional[float],
                      debtors: Optional[float]) -> Optional[float]:
    """The real ratio, for companies where both figures were filed.

    Feed this back through `cli.py calibrate` to replace the default with a
    number derived from your own list rather than a round guess.
    """
    if not turnover or not debtors or debtors <= 0:
        return None
    ratio = turnover / debtors
    low, high = PLAUSIBLE_RATIO
    if not low <= ratio <= high:
        return None          # one of the two figures was misread
    return ratio