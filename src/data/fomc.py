"""
FOMC meeting-calendar scraper (raw dates only).

Produces a flat table with columns ``date``, ``is_fomc_day`` (policy-decision
day) and ``is_fomc_press_conference``. It combines two sources:
  * historical archive pages for 2000-2020
    (federalreserve.gov/monetarypolicy/fomchistorical{year}.htm)
  * the current calendar page for 2021+
    (federalreserve.gov/monetarypolicy/fomccalendars.htm)

RAW only: the derived ``days_since_fomc`` feature depends on the trading
calendar, so it is computed later in the feature layer (a merge_asof against the
price panel's index), not here.

`requests` and `bs4` are imported lazily so the module imports without them.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from time import sleep
from typing import Optional

import pandas as pd

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
    "December": 12,
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

_COLUMNS = ["date", "is_fomc_day", "is_fomc_press_conference"]


def parse_meeting_title(title: str, year: int) -> Optional[list[dict]]:
    """Parse a meeting heading into one or two calendar rows.

    Handles: ``March 21``, ``January 30-31``, ``July 31-August 1`` and
    ``Jul/Aug 31-1``. For two-day meetings the decision (second day) carries
    ``is_fomc_day=1``. Returns None for an unrecognised format.
    """
    title = re.sub(r"\s*\([^)]*\)", "", title.strip()).strip()

    # Case 1: "March 21"
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})", title)
    if m:
        month, day = MONTHS[m.group(1)], int(m.group(2))
        return [{
            "date": datetime(year, month, day).strftime("%Y-%m-%d"),
            "is_fomc_day": 1,
            "is_fomc_press_conference": 1,
        }]

    # Case 2: "January 30-31"
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})-(\d{1,2})", title)
    if m:
        month = MONTHS[m.group(1)]
        first = datetime(year, month, int(m.group(2)))
        second = datetime(year, month, int(m.group(3)))
        return [
            {"date": first.strftime("%Y-%m-%d"),
             "is_fomc_day": 0, "is_fomc_press_conference": 1},
            {"date": second.strftime("%Y-%m-%d"),
             "is_fomc_day": 1, "is_fomc_press_conference": 0},
        ]

    # Case 3: "July 31-August 1"
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})-([A-Za-z]+)\s+(\d{1,2})", title)
    if m:
        first = datetime(year, MONTHS[m.group(1)], int(m.group(2)))
        second = datetime(year, MONTHS[m.group(3)], int(m.group(4)))
        return [
            {"date": first.strftime("%Y-%m-%d"),
             "is_fomc_day": 0, "is_fomc_press_conference": 1},
            {"date": second.strftime("%Y-%m-%d"),
             "is_fomc_day": 1, "is_fomc_press_conference": 0},
        ]

    # Case 4: "Jul/Aug 31-1"
    m = re.fullmatch(r"([A-Za-z]+)/([A-Za-z]+)\s+(\d{1,2})-(\d{1,2})", title)
    if m:
        first = datetime(year, MONTHS[m.group(1)], int(m.group(3)))
        second = datetime(year, MONTHS[m.group(2)], int(m.group(4)))
        return [
            {"date": first.strftime("%Y-%m-%d"),
             "is_fomc_day": 0, "is_fomc_press_conference": 1},
            {"date": second.strftime("%Y-%m-%d"),
             "is_fomc_day": 1, "is_fomc_press_conference": 0},
        ]

    return None


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMNS)


def scrape_fomc_historical_year(year: int) -> pd.DataFrame:
    """Scrape one historical archive page (2000-2020)."""
    import requests
    from bs4 import BeautifulSoup

    url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    article = soup.find("div", id="article")

    rows: list[dict] = []
    for panel in article.select("div.panel.panel-default"):
        h5 = panel.find("h5")
        if h5 is None:
            continue
        heading = h5.get_text(" ", strip=True)

        # Only meetings with a policy Statement; skip notation votes.
        has_statement = any(
            "statement" in a.get_text(strip=True).lower()
            for a in panel.find_all("a")
        )
        if not has_statement or "notation vote" in heading.lower():
            continue

        title = re.sub(r"\s*-\s*\d{4}$", "", heading).replace(" Meeting", "").strip()
        parsed = parse_meeting_title(title, year)
        if parsed:
            rows.extend(parsed)
        else:
            print(f"[fomc] skipping unknown format: {heading}")

    if not rows:
        return _empty()
    return (
        pd.DataFrame(rows).drop_duplicates().sort_values("date").reset_index(drop=True)
    )


def scrape_fomc_recent() -> pd.DataFrame:
    """Scrape completed meetings (2021+) from the current calendar page."""
    import requests
    from bs4 import BeautifulSoup

    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    rows: list[dict] = []
    for panel in soup.select("div.panel.panel-default"):
        h4 = panel.find("h4")
        if h4 is None or not re.match(r"\d{4} FOMC Meetings", h4.get_text(" ", strip=True)):
            continue

        for meeting in panel.select("div.fomc-meeting"):
            # Identify the policy statement by its href pattern, not link text.
            statement_link = meeting.find("a", href=re.compile(r"monetary\d{8}a"))
            if statement_link is None:
                continue  # future / malformed meeting

            m = re.search(r"(\d{8})", statement_link.get("href", ""))
            if m is None:
                continue
            decision = datetime.strptime(m.group(1), "%Y%m%d")

            date_text = (
                meeting.select_one(".fomc-meeting__date").get_text(strip=True).replace("*", "")
            )
            if "-" in date_text:  # two-day meeting
                conference = decision - timedelta(days=1)
                rows.append({"date": conference.strftime("%Y-%m-%d"),
                             "is_fomc_day": 0, "is_fomc_press_conference": 1})
                rows.append({"date": decision.strftime("%Y-%m-%d"),
                             "is_fomc_day": 1, "is_fomc_press_conference": 0})
            else:
                rows.append({"date": decision.strftime("%Y-%m-%d"),
                             "is_fomc_day": 1, "is_fomc_press_conference": 1})

    if not rows:
        return _empty()
    return (
        pd.DataFrame(rows).drop_duplicates().sort_values("date").reset_index(drop=True)
    )


def scrape_fomc_calendar(
    start_year: int = 2000,
    end_year: int = 2020,
    *,
    include_recent: bool = True,
    pause: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Scrape the full FOMC calendar: historical years plus the recent page.

    `start_year`..`end_year` cover the historical archive pages; `include_recent`
    additionally scrapes 2021+ from the live calendar page.
    """
    frames = []
    for year in range(start_year, end_year + 1):
        if verbose:
            print(f"[fomc] scraping historical {year} ...")
        frames.append(scrape_fomc_historical_year(year))
        sleep(pause)

    if include_recent:
        if verbose:
            print("[fomc] scraping recent (2021+) calendar page ...")
        frames.append(scrape_fomc_recent())

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )
