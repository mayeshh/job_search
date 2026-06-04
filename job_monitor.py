#!/usr/bin/env python3
"""
Job Board Monitor
─────────────────
Polls company career pages, filters new listings by keyword/location rules,
and notifies via console and email.

Architecture
────────────
                              ┌──────────────────┐
                              │   JobMonitor     │
                              └────────┬─────────┘
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
      ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
      │ APISources   │         │   Browser    │         │  Notifiers   │
      │ (concurrent) │         │   Sources    │         │              │
      └──────────────┘         │ (sequential, │         └──────────────┘
       Greenhouse direct        │  shared Chrome)        Console / Email
       Lever direct             └──────────────┘
       SmartRecruiters direct    Kite / Gilead / Labcorp / AstraZeneca
                                 (Workday boards)

API sources hit the ATS directly (fast). Browser sources load the company's
own career page in headless Chrome, then intercept the XHR response the page
makes to its ATS — same JSON, but the request originates from an allowlisted
origin so it isn't rejected. The interception is robust to DOM changes.

Setup
─────
1.  Install dependencies:
        pip install requests playwright python-dotenv
        playwright install chromium

2.  Create a .env file next to this script:
        JOB_MONITOR_EMAIL=you@gmail.com
        JOB_MONITOR_PASSWORD=xxxx-xxxx-xxxx-xxxx   ← Gmail App Password (NOT your login password)

    To get a Gmail App Password:
        • Visit https://myaccount.google.com/security
        • Enable 2-Step Verification (required)
        • Search for "App passwords" → create one for Mail
        • Paste the 16-character code as JOB_MONITOR_PASSWORD

Run once
────────
    python job_monitor.py

Cron (daily at 8 AM)
────────────────────
    crontab -e          ← opens your crontab in $EDITOR

    Add this line (update paths to match your environment):
        0 8 * * * /opt/anaconda3/envs/job_search/bin/python /Users/mayasegal/Documents/personal/job_search/code/job_search/job_monitor.py >> /tmp/job_monitor.log 2>&1

    View the log:
        cat /tmp/job_monitor.log

    Verify the cron entry saved:
        crontab -l
"""

from __future__ import annotations

import json
import logging
import re
import smtplib
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator, Pattern
import os 
import requests
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# ─── LOGGING ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(name)-30s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("monitor")


# ─── UTILITIES ───────────────────────────────────────────────────────────────

def retry(
    attempts: int = 3,
    backoff: float = 1.5,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Retry a function with exponential backoff on transient exceptions."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            delay = 1.0
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == attempts:
                        raise
                    log.warning(f"{fn.__name__} failed ({e}); retry {attempt}/{attempts - 1} in {delay:.1f}s")
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator


HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Mozilla/5.0 (compatible; job-monitor/1.0)"})


# ─── DOMAIN MODEL ────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Job:
    """Normalized job listing — every source maps to this shape."""
    source:   str
    title:    str
    location: str
    url:      str
    job_id:   str

    @property
    def cache_key(self) -> str:
        return f"{self.source}::{self.job_id}"


@dataclass
class JobFilter:
    """Composable filter applied uniformly to all sources."""
    keywords:          tuple[str, ...] = ()
    remote_only:       bool            = False
    location_contains: str | None      = None
    location_any:      tuple[str, ...] = ()  # OR logic: passes if ANY term matches location

    def matches(self, job: Job) -> bool:
        title_lc    = job.title.lower()
        location_lc = job.location.lower()

        if self.keywords and not any(kw.lower() in title_lc for kw in self.keywords):
            return False
        if self.remote_only and "remote" not in location_lc:
            return False
        if self.location_contains and self.location_contains.lower() not in location_lc:
            return False
        if self.location_any and not any(loc.lower() in location_lc for loc in self.location_any):
            return False
        return True


# ─── CACHE ───────────────────────────────────────────────────────────────────

class JobCache:
    """JSON-backed set of seen job cache keys. Single file, atomic write."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._seen: set[str] = set(json.loads(path.read_text())) if path.exists() else set()

    def is_new(self, job: Job) -> bool:
        return job.cache_key not in self._seen

    def mark_seen(self, job: Job) -> None:
        self._seen.add(job.cache_key)

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self._seen), indent=2))
        tmp.replace(self.path)


# ─── BROWSER CONTEXT ─────────────────────────────────────────────────────────

class BrowserContext:
    """
    Lazy-initialized shared Playwright browser. Created on first use, closed
    when the monitor finishes. Each browser source gets its own page (cheap)
    rather than its own browser instance (expensive).
    """

    def __init__(self) -> None:
        self._playwright = None
        self._browser    = None

    def _ensure_started(self) -> None:
        if self._browser is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            ) from e

        log.info("launching headless chromium")
        self._playwright = sync_playwright().start()
        self._browser    = self._playwright.chromium.launch(headless=True)

    @contextmanager
    def page(self, timeout_ms: int = 30_000):
        """Yield a fresh page with a sane default timeout."""
        self._ensure_started()
        ctx  = self._browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        )
        page = ctx.new_page()
        page.set_default_timeout(timeout_ms)
        try:
            yield page
        finally:
            ctx.close()

    def close(self) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()


# ─── SOURCE BASE ─────────────────────────────────────────────────────────────

class JobSource(ABC):
    """
    Abstract base. Subclasses implement fetch_raw() and parse_jobs();
    the template method fetch() handles logging and exception isolation.
    """

    kind: ClassVar[str] = "abstract"

    def __init__(self, name: str, filter_: JobFilter | None = None) -> None:
        self.name    = name
        self.filter_ = filter_ or JobFilter()
        self.log     = logging.getLogger(f"src.{self.kind}.{name}")

    @abstractmethod
    def fetch_raw(self) -> Any: ...

    @abstractmethod
    def parse_jobs(self, raw: Any) -> Iterator[Job]: ...

    def fetch(self) -> list[Job]:
        try:
            raw  = self.fetch_raw()
            jobs = [j for j in self.parse_jobs(raw) if self.filter_.matches(j)]
            self.log.info(f"{len(jobs)} matching listing(s)")
            return jobs
        except Exception as e:
            self.log.error(f"fetch failed: {e}")
            return []


# ─── API SOURCES (direct HTTP, no browser) ───────────────────────────────────

class GreenhouseSource(JobSource):
    """Greenhouse public API for boards without origin allowlists."""
    kind = "greenhouse"
    API  = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"

    def __init__(self, name: str, slug: str, filter_: JobFilter | None = None) -> None:
        super().__init__(name, filter_)
        self.slug = slug

    @retry(exceptions=(requests.RequestException,))
    def fetch_raw(self) -> dict:
        r = HTTP.get(self.API.format(slug=self.slug), timeout=10)
        r.raise_for_status()
        return r.json()

    def parse_jobs(self, raw: dict) -> Iterator[Job]:
        yield from _parse_greenhouse_payload(raw, self.name)


class LeverSource(JobSource):
    """Lever public API for boards without origin allowlists."""
    kind = "lever"
    API  = "https://api.lever.co/v0/postings/{company}?mode=json"

    def __init__(self, name: str, company: str, filter_: JobFilter | None = None) -> None:
        super().__init__(name, filter_)
        self.company = company

    @retry(exceptions=(requests.RequestException,))
    def fetch_raw(self) -> list[dict]:
        r = HTTP.get(self.API.format(company=self.company), timeout=10)
        r.raise_for_status()
        return r.json()

    def parse_jobs(self, raw: list[dict]) -> Iterator[Job]:
        yield from _parse_lever_payload(raw, self.name)


class SmartRecruiterSource(JobSource):
    """SmartRecruiters public postings API (Guardant Health, AbbVie, etc.)."""
    kind = "smartrecruiters"
    API  = "https://api.smartrecruiters.com/v1/companies/{company}/postings"

    def __init__(self, name: str, company: str, filter_: JobFilter | None = None) -> None:
        super().__init__(name, filter_)
        self.company = company

    @retry(exceptions=(requests.RequestException,))
    def fetch_raw(self) -> dict:
        r = HTTP.get(self.API.format(company=self.company), timeout=10)
        r.raise_for_status()
        return r.json()

    def parse_jobs(self, raw: dict) -> Iterator[Job]:
        for job in raw.get("content", []):
            loc = job.get("location", {})
            if loc.get("remote"):
                location = "Remote"
            else:
                parts    = [loc.get("city", ""), loc.get("region", ""), loc.get("country", "")]
                location = ", ".join(p for p in parts if p) or "N/A"
            yield Job(
                source   = self.name,
                title    = job.get("name", ""),
                location = location,
                url      = job.get("ref", ""),
                job_id   = job.get("id", ""),
            )


# ─── BROWSER SOURCES (use Playwright to bypass allowlists) ───────────────────

class BrowserSource(JobSource):
    """
    Base for browser-based scraping. Subclasses provide:
      - api_pattern : regex matching the ATS API URL the page calls
      - parse_jobs  : how to read the intercepted JSON

    fetch_raw() loads the careers page, captures the first matching XHR/fetch
    response, and returns its parsed JSON body.
    """

    api_pattern: ClassVar[Pattern[str]]

    def __init__(
        self,
        name:        str,
        url:         str,
        browser:     BrowserContext,
        filter_:     JobFilter | None = None,
        wait_state:  str  = "networkidle",
        timeout_ms:  int  = 30_000,
    ) -> None:
        super().__init__(name, filter_)
        self.url        = url
        self.browser    = browser
        self.wait_state = wait_state
        self.timeout_ms = timeout_ms

    @retry(attempts=2, exceptions=(TimeoutError, RuntimeError))
    def fetch_raw(self) -> Any:
        captured: list[Any] = []

        def on_response(response):
            try:
                if self.api_pattern.search(response.url) and response.ok:
                    captured.append(response.json())
            except Exception:
                pass  # not all responses are JSON

        with self.browser.page(self.timeout_ms) as page:
            page.on("response", on_response)
            page.goto(self.url, wait_until=self.wait_state)
            # Give late XHRs a moment to land
            page.wait_for_timeout(1500)

        if not captured:
            raise RuntimeError(f"no response matching {self.api_pattern.pattern} captured at {self.url}")
        return captured[-1]  # last response usually has the full result set


class GreenhouseBrowserSource(BrowserSource):
    """
    For Greenhouse boards with origin allowlists (Guardant, Grail, Myriad, etc.).
    Point at the company's actual careers URL — the embedded Greenhouse iframe
    will call the API from the allowlisted origin and we capture the response.
    """
    kind        = "greenhouse_browser"
    api_pattern = re.compile(r"boards-api\.greenhouse\.io/.*?/jobs")

    def parse_jobs(self, raw: dict) -> Iterator[Job]:
        yield from _parse_greenhouse_payload(raw, self.name)


class WorkdayBrowserSource(BrowserSource):
    """
    For Workday-hosted career sites (Gilead, Amgen, AbbVie, etc.).
    Point at the public job board URL; we intercept the wday/cxs API response.
    """
    kind        = "workday_browser"
    api_pattern = re.compile(r"/wday/cxs/.*?/jobs")

    def parse_jobs(self, raw: dict) -> Iterator[Job]:
        base = re.match(r"https?://[^/]+", self.url).group(0)
        for job in raw.get("jobPostings", []):
            external = job.get("externalPath", "")
            yield Job(
                source   = self.name,
                title    = job.get("title", ""),
                location = job.get("locationsText", "N/A"),
                url      = f"{base}{external}" if external else self.url,
                job_id   = external or job.get("title", ""),
            )


class LeverBrowserSource(BrowserSource):
    """For Lever boards with origin restrictions."""
    kind        = "lever_browser"
    api_pattern = re.compile(r"api\.lever\.co/v0/postings/")

    def parse_jobs(self, raw: list[dict]) -> Iterator[Job]:
        yield from _parse_lever_payload(raw, self.name)


# ─── SHARED PARSERS ──────────────────────────────────────────────────────────

def _parse_greenhouse_payload(raw: dict, source_name: str) -> Iterator[Job]:
    for job in raw.get("jobs", []):
        yield Job(
            source   = source_name,
            title    = job.get("title", ""),
            location = job.get("location", {}).get("name", "N/A"),
            url      = job.get("absolute_url", ""),
            job_id   = str(job["id"]),
        )


def _parse_lever_payload(raw: list[dict], source_name: str) -> Iterator[Job]:
    for job in raw:
        yield Job(
            source   = source_name,
            title    = job.get("text", ""),
            location = job.get("categories", {}).get("location", "N/A"),
            url      = job.get("hostedUrl", ""),
            job_id   = job.get("id", ""),
        )


# ─── NOTIFIERS ───────────────────────────────────────────────────────────────

class Notifier(ABC):
    @abstractmethod
    def notify(self, jobs: list[Job]) -> None: ...


class ConsoleNotifier(Notifier):
    def notify(self, jobs: list[Job]) -> None:
        if not jobs:
            log.info("no new listings")
            return
        print()
        for job in jobs:
            print(f"  ★ [{job.source}] {job.title}")
            print(f"    {job.location}")
            print(f"    {job.url}\n")


@dataclass
class EmailNotifier(Notifier):
    sender:    str
    password:  str
    recipient: str
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587

    def notify(self, jobs: list[Job]) -> None:
        if not jobs:
            return

        date    = datetime.now().strftime("%b %d, %Y")
        subject = f"[Job Monitor] {len(jobs)} new listing(s) — {date}"
        body    = self._format_body(jobs)

        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = self.sender, self.recipient, subject
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as s:
                s.starttls()
                s.login(self.sender, self.password)
                s.send_message(msg)
            log.info(f"emailed {len(jobs)} listing(s) → {self.recipient}")
        except Exception as e:
            log.error(f"email send failed: {e}")

    @staticmethod
    def _format_body(jobs: list[Job]) -> str:
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"New job listings — {ts}", ""]
        by_source: dict[str, list[Job]] = {}
        for job in jobs:
            by_source.setdefault(job.source, []).append(job)
        for source, source_jobs in by_source.items():
            lines += [f"━━━ {source} ━━━"]
            for job in source_jobs:
                lines += [f"  • {job.title}", f"    {job.location}", f"    {job.url}", ""]
        return "\n".join(lines)


# ─── ORCHESTRATOR ────────────────────────────────────────────────────────────

@dataclass
class JobMonitor:
    """
    Runs all sources, dedupes against cache, dispatches notifications.

    Execution model:
      • API sources run in parallel (ThreadPoolExecutor, I/O bound)
      • Browser sources run sequentially in the shared browser
        (sync Playwright isn't thread-safe; parallel browsers are memory-heavy)
    """
    api_sources:     list[JobSource]
    browser_sources: list[BrowserSource]
    cache:           JobCache
    notifiers:       list[Notifier]
    browser:         BrowserContext
    workers:         int = 8

    def run(self) -> list[Job]:
        total = len(self.api_sources) + len(self.browser_sources)
        log.info(f"polling {total} sources ({len(self.api_sources)} API, {len(self.browser_sources)} browser)")

        all_jobs: list[Job] = []
        all_jobs.extend(self._run_api_sources())
        all_jobs.extend(self._run_browser_sources())

        new_jobs = [j for j in all_jobs if self.cache.is_new(j)]
        for job in new_jobs:
            self.cache.mark_seen(job)
        self.cache.save()

        log.info(f"{len(new_jobs)} new of {len(all_jobs)} total matching listings")
        for notifier in self.notifiers:
            notifier.notify(new_jobs)

        self.browser.close()
        return new_jobs

    def _run_api_sources(self) -> list[Job]:
        if not self.api_sources:
            return []
        jobs: list[Job] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(src.fetch) for src in self.api_sources]
            for future in as_completed(futures):
                jobs.extend(future.result())
        return jobs

    def _run_browser_sources(self) -> list[Job]:
        jobs: list[Job] = []
        for src in self.browser_sources:
            jobs.extend(src.fetch())
        return jobs


# ─── CONFIG ──────────────────────────────────────────────────────────────────

CACHE_PATH = Path.home() / ".job_monitor_cache.json"
BROWSER    = BrowserContext()

EMAIL = EmailNotifier(
    sender    = os.environ["JOB_MONITOR_EMAIL"],
    password  = os.environ["JOB_MONITOR_PASSWORD"],
    recipient = os.environ["JOB_MONITOR_EMAIL"],
)

DEFAULT_FILTER = JobFilter(
    keywords = (
        "bioinformatics",
        "bioinformatician",
        "biophysics",
        "computational biolog",
        "genomics",
        "epigenomics",
        "sequencing",
        "oncology",
        "cancer",
        "data scientist",
        "machine learning",
        "methylation",
        "pipeline",
        "scientist",
    ),
)
REMOTE_ONLY     = replace(DEFAULT_FILTER, remote_only=True)
CALIFORNIA_ONLY = replace(DEFAULT_FILTER, location_contains="california")
# Passes remote jobs (any geography) OR local CA jobs (in-person, hybrid, or remote).
# ", ca" catches "Santa Monica, CA"-style locations that omit the full state name.
LOCAL_OR_REMOTE = replace(DEFAULT_FILTER, location_any=("remote", "california", ", ca"))


# ─── SOURCES ─────────────────────────────────────────────────────────────────
# To add a company:
#   1. Find its careers URL (where the public job board lives)
#   2. Pick a source class based on its ATS (Greenhouse / Workday / Lever / etc.)
#   3. Add a one-liner below

API_SOURCES: list[JobSource] = [
    # Greenhouse (direct API)
    GreenhouseSource("Natera",           "natera",           LOCAL_OR_REMOTE),
    GreenhouseSource("Tempus",           "tempus",           LOCAL_OR_REMOTE),
    GreenhouseSource("Parse Biosciences","parsebiosciences",  LOCAL_OR_REMOTE),
    GreenhouseSource("10x Genomics",     "10xgenomics",      LOCAL_OR_REMOTE),
    GreenhouseSource("BridgeBio",        "bridgebio",        LOCAL_OR_REMOTE),
    GreenhouseSource("Triplebar",        "triplebarbio",     LOCAL_OR_REMOTE),
    # Astera Labs is a semiconductor/AI-infrastructure company — verify this is the right "Astera"
    GreenhouseSource("Astera Labs",      "asteralabs",       LOCAL_OR_REMOTE),

    # Lever (direct API)
    LeverSource("Grail",                 "grailbio",         LOCAL_OR_REMOTE),

    # SmartRecruiters (direct API)
    SmartRecruiterSource("Guardant Health", "GuardantHealth", LOCAL_OR_REMOTE),
    SmartRecruiterSource("AbbVie",          "abbvie",         LOCAL_OR_REMOTE),
]

BROWSER_SOURCES: list[BrowserSource] = [
    # Workday sites (use the direct *.myworkdayjobs.com URL, not the wrapper careers page)
    WorkdayBrowserSource("Gilead",        "https://gilead.wd1.myworkdayjobs.com/en-US/gileadcareers",       BROWSER, LOCAL_OR_REMOTE),
    WorkdayBrowserSource("Kite Pharma",   "https://gilead.wd1.myworkdayjobs.com/kitepharmacareers",         BROWSER, LOCAL_OR_REMOTE),
    WorkdayBrowserSource("Amgen",         "https://amgen.wd1.myworkdayjobs.com/en-US/Careers",              BROWSER, LOCAL_OR_REMOTE),
    WorkdayBrowserSource("Thermo Fisher", "https://thermofisher.wd5.myworkdayjobs.com/ThermoFisherCareers", BROWSER, LOCAL_OR_REMOTE),
    WorkdayBrowserSource("Labcorp",       "https://labcorp.wd1.myworkdayjobs.com/External",                 BROWSER, LOCAL_OR_REMOTE),
    # AstraZeneca — Santa Monica is a primary US site; LOCAL_OR_REMOTE catches it via ", ca"
    WorkdayBrowserSource("AstraZeneca",   "https://astrazeneca.wd3.myworkdayjobs.com/Careers",              BROWSER, LOCAL_OR_REMOTE),

    # TODO: Myriad Genetics — uses Oracle Cloud HCM, not yet supported; check myriad.com/careers manually
    # TODO: Personalis — ATS undetectable from page HTML; check personalis.com/about/careers manually
    # TODO: Karius — no public ATS board found; check kariusdx.com/about-us/careers manually
    # TODO: Ambry Genetics — uses UltiPro ATS, not yet supported; check ambrygen.com/company/careers manually
    # TODO: Evozyne — no standard ATS detected; check evozyne.com/careers manually
]


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main() -> None:
    JobMonitor(
        api_sources     = API_SOURCES,
        browser_sources = BROWSER_SOURCES,
        cache           = JobCache(CACHE_PATH),
        notifiers       = [ConsoleNotifier(), EMAIL],
        browser         = BROWSER,
    ).run()


if __name__ == "__main__":
    main()