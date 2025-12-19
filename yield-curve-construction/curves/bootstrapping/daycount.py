from datetime import datetime

def yearfrac(d0: datetime, d1: datetime, basis: str = "ACT/365.25") -> float:
    days = (d1 - d0).days
    if basis.upper() == "ACT/360":
        return days / 360.0
    if basis.upper() == "ACT/365":
        return days / 365.0
    return days / 365.25
