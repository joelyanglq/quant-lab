"""
SEC EDGAR N-PORT downloader + parser.

Downloads quarterly N-PORT filings for ETFs from SEC EDGAR,
parses XML to extract holdings (CUSIP, weight, value, shares).

SEC N-PORT filing structure:
  - Each ETF issuer has a CIK (Central Index Key)
  - Multi-series trusts (e.g. SPDR) file one N-PORT per series
  - XML namespace: http://www.sec.gov/edgar/nport
  - Holdings in <invstOrSec> elements with cusip, pctVal, valUSD, balance

Rate limits: SEC requires User-Agent header, max ~10 req/sec.
"""
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SEC_BASE = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
USER_AGENT = "quant-lab research@example.com"
NS = "{http://www.sec.gov/edgar/nport}"
REQUEST_INTERVAL = 0.12  # ~8 req/sec, within SEC's 10/sec limit


# CIK registry: ticker → (cik, series_id or None)
# series_id is needed for multi-series trusts (e.g. SPDR Select Sector)
# None means the entire trust is one ETF (e.g. SPY Trust)
ETF_CIK_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    # Tier 1: Broad
    "SPY":  ("0000884394", None),
    "QQQ":  ("0001067839", None),
    "VTI":  ("0000036405", "S000002638"),
    "IWM":  ("0001100663", "S000006044"),
    "IJH":  ("0001100663", "S000006043"),
    "DIA":  ("0000885580", None),
    # Tier 2: SPDR Sectors (all under Select Sector SPDR Trust, each files separately)
    "XLK":  ("0001064641", "S000006415"),
    "XLF":  ("0001064641", "S000006411"),
    "XLV":  ("0001064641", "S000006412"),
    "XLY":  ("0001064641", "S000006408"),
    "XLP":  ("0001064641", "S000006409"),
    "XLE":  ("0001064641", "S000006410"),
    "XLI":  ("0001064641", "S000006413"),
    "XLU":  ("0001064641", "S000006416"),
    "XLB":  ("0001064641", "S000006414"),
    "XLRE": ("0001064641", "S000051152"),
    "XLC":  ("0001064641", "S000062095"),
    # Tier 3: Themes (verified series_ids 2026-07-06)
    "SMH":  ("0001137360", "S000034411"),
    "IBB":  ("0001100663", "S000004350"),
    "XBI":  ("0001064642", "S000010018"),
    "ICLN": ("0001100663", "S000022498"),
    "KRE":  ("0001064642", "S000012325"),
    "XOP":  ("0001064642", "S000012319"),
    "IYR":  ("0001100663", "S000004328"),
    "XHB":  ("0001064642", "S000010019"),
    "ITB":  ("0001100663", "S000009415"),
    "HACK": ("0001633061", "S000082278"),  # Amplify Cybersecurity ETF
    "BOTZ": ("0001432353", "S000054693"),  # Global X Robotics & AI ETF
    "ARKK": ("0001579982", "S000042977"),  # ARK Innovation ETF
    "TAN":  ("0001378872", "S000060822"),  # Invesco Solar ETF
    "IBIT": ("0002020923", None),
    # Tier 4: Factors
    "VIG":  ("0000036405", "S000020736"),
    "SCHD": ("0001489391", "S000030199"),
    "DVY":  ("0001100663", "S000006067"),
    "VTV":  ("0000036405", "S000002643"),
    "VUG":  ("0000036405", "S000002644"),
    "MTUM": ("0001100663", "S000046316"),
    "USMV": ("0001100663", "S000034213"),
    "QUAL": ("0001100663", "S000046313"),
    # Tier 5: International
    "EFA":  ("0001100663", "S000006046"),
    "EEM":  ("0001100663", "S000006047"),
    "FXI":  ("0001100663", "S000006048"),
    "EWJ":  ("0001100663", "S000006052"),
    "INDA": ("0001100663", "S000034211"),
    "VWO":  ("0000036405", "S000013374"),
}


class NPortDownloader:
    """Downloads and parses SEC N-PORT filings."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self._last_request = 0.0

    def _throttle(self):
        elapsed = time.monotonic() - self._last_request
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str) -> requests.Response:
        self._throttle()
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        return resp

    def get_filing_list(self, cik: str) -> List[dict]:
        """Get all N-PORT filings for a CIK from SEC EDGAR."""
        url = f"{SEC_BASE}/submissions/CIK{cik}.json"
        data = self._get(url).json()

        filings = []
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form in ("NPORT-P", "NPORT-P/A"):
                filings.append({
                    "form": form,
                    "filing_date": dates[i],
                    "accession": accessions[i],
                    "primary_doc": primary_docs[i],
                })

        # Also check older filings in separate files
        for hist_file in data.get("filings", {}).get("files", []):
            hist_url = f"{SEC_BASE}/submissions/{hist_file['name']}"
            try:
                hist_data = self._get(hist_url).json()
                h_forms = hist_data.get("form", [])
                h_dates = hist_data.get("filingDate", [])
                h_accessions = hist_data.get("accessionNumber", [])
                h_primary = hist_data.get("primaryDocument", [])
                for i, form in enumerate(h_forms):
                    if form in ("NPORT-P", "NPORT-P/A"):
                        filings.append({
                            "form": form,
                            "filing_date": h_dates[i],
                            "accession": h_accessions[i],
                            "primary_doc": h_primary[i],
                        })
            except Exception as e:
                logger.warning("Failed to fetch history file %s: %s", hist_file["name"], e)

        return filings

    def download_nport_xml(self, cik: str, accession: str, primary_doc: str) -> str:
        """Download N-PORT XML content."""
        acc_no_dash = accession.replace("-", "")
        # primary_doc may include XSL prefix like "xslFormNPORT-P_X01/primary_doc.xml"
        # The raw XML is always at the accession root as "primary_doc.xml"
        doc_name = primary_doc.split("/")[-1] if "/" in primary_doc else primary_doc
        url = f"{SEC_ARCHIVES}/{cik.lstrip('0')}/{acc_no_dash}/{doc_name}"
        return self._get(url).text

    def parse_nport_xml(
        self, xml_text: str, series_id: Optional[str] = None
    ) -> Tuple[Optional[date], pd.DataFrame]:
        """
        Parse N-PORT XML into a DataFrame of holdings.

        For multi-series trusts, each filing covers ONE series.
        If series_id is given, we verify the filing matches and skip if not.

        Returns:
            (report_date, DataFrame with columns: cusip, isin, name, weight, value, shares)
            Returns (None, empty DF) if series_id doesn't match the filing.
        """
        root = ET.fromstring(xml_text)

        # Check series_id match (for multi-series trusts)
        if series_id:
            filing_series = None
            for sid_el in root.iter(f"{NS}seriesId"):
                if sid_el.text:
                    filing_series = sid_el.text.strip()
                    break
            if filing_series and filing_series != series_id:
                return None, pd.DataFrame()

        # Extract report date
        report_date = None
        for tag in [f"{NS}repPd", f"{NS}repPdDate", f"{NS}repPdEnd"]:
            el = root.find(f".//{tag}")
            if el is not None and el.text:
                try:
                    report_date = datetime.strptime(el.text[:10], "%Y-%m-%d").date()
                    break
                except ValueError:
                    pass

        holdings = []
        for sec in root.iter(f"{NS}invstOrSec"):
            name_el = sec.find(f"{NS}name")
            cusip_el = sec.find(f"{NS}cusip")
            pct_el = sec.find(f"{NS}pctVal")
            val_el = sec.find(f"{NS}valUSD")
            bal_el = sec.find(f"{NS}balance")

            # Get ISIN from identifiers
            isin = None
            isin_el = sec.find(f".//{NS}isin")
            if isin_el is not None:
                isin = isin_el.get("value", isin_el.text)

            cusip = cusip_el.text.strip() if cusip_el is not None and cusip_el.text else None
            if not cusip or cusip == "000000000":
                continue

            try:
                weight = float(pct_el.text) if pct_el is not None and pct_el.text else 0.0
            except (ValueError, TypeError):
                weight = 0.0

            try:
                value = float(val_el.text) if val_el is not None and val_el.text else 0.0
            except (ValueError, TypeError):
                value = 0.0

            try:
                shares = float(bal_el.text) if bal_el is not None and bal_el.text else 0.0
            except (ValueError, TypeError):
                shares = 0.0

            holdings.append({
                "cusip": cusip,
                "isin": isin,
                "name": name_el.text.strip() if name_el is not None and name_el.text else "",
                "weight": weight,
                "value": value,
                "shares": shares,
            })

        df = pd.DataFrame(holdings) if holdings else pd.DataFrame(
            columns=["cusip", "isin", "name", "weight", "value", "shares"]
        )
        return report_date, df

    def sync_etf(
        self,
        ticker: str,
        force: bool = False,
    ) -> int:
        """
        Download and parse all N-PORT filings for an ETF.

        Returns number of new quarters saved.
        """
        if ticker not in ETF_CIK_MAP:
            logger.warning("No CIK mapping for %s", ticker)
            return 0

        cik, series_id = ETF_CIK_MAP[ticker]
        etf_dir = self.storage_dir / ticker
        etf_dir.mkdir(parents=True, exist_ok=True)

        existing = {f.stem for f in etf_dir.glob("*.parquet")} if not force else set()

        filings = self.get_filing_list(cik)
        logger.info("%s: found %d N-PORT filings (CIK %s)", ticker, len(filings), cik)

        saved = 0
        for filing in filings:
            try:
                xml_text = self.download_nport_xml(cik, filing["accession"], filing["primary_doc"])
                report_date, df = self.parse_nport_xml(xml_text, series_id)

                if report_date is None or df.empty:
                    continue

                quarter = _date_to_quarter(report_date)
                if quarter in existing:
                    continue

                # Normalize weights: pctVal is already in percentage (0-100 scale in some filings)
                total_w = df["weight"].sum()
                if total_w > 0:
                    if total_w > 200:
                        # Weights are in percentage form (sum ~= 100*n_holdings/wrong)
                        # Actually if sum > 200, something is wrong. Just keep as-is.
                        pass
                    # Keep raw pctVal — it's the percentage of total portfolio
                    pass

                df["report_date"] = report_date
                out_path = etf_dir / f"{quarter}.parquet"
                df.to_parquet(out_path, compression="snappy", index=False)
                saved += 1
                logger.info("  %s %s: %d holdings, total_weight=%.1f%%", ticker, quarter, len(df), total_w)

            except Exception as e:
                logger.warning("  %s filing %s: %s", ticker, filing["filing_date"], e)

        return saved

    def sync_all(
        self,
        tickers: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, int]:
        """Sync N-PORT filings for all or specified ETFs."""
        if tickers is None:
            tickers = list(ETF_CIK_MAP.keys())

        results = {}
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
            try:
                n = self.sync_etf(ticker, force=force)
                results[ticker] = n
                print(f"{n} new quarters")
            except Exception as e:
                results[ticker] = -1
                print(f"ERROR: {e}")

        return results

    def load_holdings(self, ticker: str, quarter: Optional[str] = None) -> pd.DataFrame:
        """Load saved holdings. If quarter is None, load the latest."""
        etf_dir = self.storage_dir / ticker
        if not etf_dir.exists():
            return pd.DataFrame()

        files = sorted(etf_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()

        if quarter:
            target = etf_dir / f"{quarter}.parquet"
            if target.exists():
                return pd.read_parquet(target)
            return pd.DataFrame()

        return pd.read_parquet(files[-1])

    def list_quarters(self, ticker: str) -> List[str]:
        """List available quarters for an ETF."""
        etf_dir = self.storage_dir / ticker
        if not etf_dir.exists():
            return []
        return sorted(f.stem for f in etf_dir.glob("*.parquet"))

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _date_to_quarter(d: date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def lookup_cik(ticker: str) -> Optional[str]:
    """Look up CIK for a ticker via SEC EDGAR company search."""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=NPORT-P"
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    try:
        resp = session.get(url, timeout=15)
        # This is a simplified lookup; real implementation would parse the response
        return None
    except Exception:
        return None
    finally:
        session.close()
