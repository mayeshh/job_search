#!/usr/bin/env python3
"""
Job Board Monitor
─────────────────
Polls configured ATS (Applicant Tracking System) endpoints, filters new
listings by keyword/location rules, and notifies via console and email.

Architecture
────────────
                         ┌──────────────────┐
                         │   JobMonitor     │ orchestrator
                         └────────┬─────────┘
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
            ┌──────────┐    ┌──────────┐   ┌────────────┐
            │  Source  │    │  Filter  │   │  Notifier  │
            │  (ABC)   │    └──────────┘   │   (ABC)    │
            └────┬─────┘                   └─────┬──────┘
       ┌────────┼────────┬─────────┐         ┌───┴───┐
       ▼        ▼        ▼         ▼         ▼       ▼
   Greenhouse Lever  Workday  IndeedRSS  Console  Email

Each source implements `fetch_raw()` and `parse_jobs()`. Adding a new ATS is
a 30-line subclass. Network I/O is parallelized with a thread pool; HTTP
sessions are reused; transient failures are retried with backoff.

Run
───
    pip install requests
    python job_monitor.py                       # one-off
    EDITOR=nano crontab -e   →   0 8 * * * /usr/bin/python3 /path/to/job_monitor.py >> /tmp/job_monitor.log 2>&1
"""

from __future__ import annotations

import json
import logging
import smtplib
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import wraps
from pathlib import Path
from typing import Callable, ClassVar, Iterable, Iterator

import requests


# ─── LOGGING ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("monitor")


# ─── UTILITIES ───────────────────────────────────────────────────────────────

def retry(
    attempts: int = 3,
    backoff: float = 1.5,
    exceptions: tuple[type[Exception], ...] = (requests.RequestException,),
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


# Shared HTTP session — connection pooling across all sources
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

    def matches(self, job: Job) -> bool:
        title_lc    = job.title.lower()
        location_lc = job.location.lower()

        if self.keywords and not any(kw.lower() in title_lc for kw in self.keywords):
            return False
        if self.remote_only and "remote" not in location_lc:
            return False
        if self.location_contains and self.location_contains.lower() not in location_lc:
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


# ─── SOURCES ─────────────────────────────────────────────────────────────────

class JobSource(ABC):
    """
    Abstract base for any ATS integration.
    Subclasses implement fetch_raw() and parse_jobs(); the template method
    fetch() handles retry, logging, and exception isolation.
    """

    kind: ClassVar[str] = "abstract"

    def __init__(self, name: str, filter_: JobFilter | None = None) -> None:
        self.name    = name
        self.filter_ = filter_ or JobFilter()
        self.log     = logging.getLogger(f"src.{self.kind}.{name}")

    @abstractmethod
    def fetch_raw(self) -> object:
        """Return the raw response payload (JSON dict, XML element, etc.)."""

    @abstractmethod
    def parse_jobs(self, raw: object) -> Iterator[Job]:
        """Yield normalized Job objects from the raw payload."""

    def fetch(self) -> list[Job]:
        """Template method. Returns filtered jobs; never raises."""
        try:
            raw  = self.fetch_raw()
            jobs = [j for j in self.parse_jobs(raw) if self.filter_.matches(j)]
            self.log.info(f"{len(jobs)} matching listing(s)")
            return jobs
        except Exception as e:
            self.log.error(f"fetch failed: {e}")
            return []


class GreenhouseSource(JobSource):
    """Greenhouse public API. Clean JSON, no auth required."""
    kind = "greenhouse"
    API  = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"

    def __init__(self, name: str, slug: str, filter_: JobFilter | None = None) -> None:
        super().__init__(name, filter_)
        self.slug = slug

    @retry()
    def fetch_raw(self) -> dict:
        r = HTTP.get(self.API.format(slug=self.slug), timeout=10)
        r.raise_for_status()
        return r.json()

    def parse_jobs(self, raw: dict) -> Iterator[Job]:
        for job in raw.get("jobs", []):
            yield Job(
                source   = self.name,
                title    = job.get("title", ""),
                location = job.get("location", {}).get("name", "N/A"),
                url      = job.get("absolute_url", ""),
                job_id   = str(job["id"]),
            )


class LeverSource(JobSource):
    """Lever public API. Same idea as Greenhouse, different schema."""
    kind = "lever"
    API  = "https://api.lever.co/v0/postings/{company}?mode=json"

    def __init__(self, name: str, company: str, filter_: JobFilter | None = None) -> None:
        super().__init__(name, filter_)
        self.company = company

    @retry()
    def fetch_raw(self) -> list[dict]:
        r = HTTP.get(self.API.format(company=self.company), timeout=10)
        r.raise_for_status()
        return r.json()

    def parse_jobs(self, raw: list[dict]) -> Iterator[Job]:
        for job in raw:
            yield Job(
                source   = self.name,
                title    = job.get("text", ""),
                location = job.get("categories", {}).get("location", "N/A"),
                url      = job.get("hostedUrl", ""),
                job_id   = job.get("id", ""),
            )


class WorkdaySource(JobSource):
    """
    Workday's internal JSON API. Not officially documented but stable.
    Endpoint pattern:
        POST https://{tenant}.{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs
    Discover tenant/cluster/board from the public job board URL.
    Example: https://gilead.wd1.myworkdayjobs.com/en-US/gileadcareers
             tenant="gilead", cluster="wd1", board="gileadcareers"
    """
    kind = "workday"
    URL  = "https://{tenant}.{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs"

    def __init__(
        self,
        name:       str,
        tenant:     str,
        board:      str,
        cluster:    str = "wd1",
        search:     str = "",
        page_limit: int = 100,
        filter_:    JobFilter | None = None,
    ) -> None:
        super().__init__(name, filter_)
        self.tenant     = tenant
        self.board      = board
        self.cluster    = cluster
        self.search     = search
        self.page_limit = page_limit

    @retry()
    def fetch_raw(self) -> dict:
        url     = self.URL.format(tenant=self.tenant, cluster=self.cluster, board=self.board)
        payload = {"limit": self.page_limit, "offset": 0, "searchText": self.search, "appliedFacets": {}}
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        r       = HTTP.post(url, json=payload, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()

    def parse_jobs(self, raw: dict) -> Iterator[Job]:
        base = f"https://{self.tenant}.{self.cluster}.myworkdayjobs.com"
        for job in raw.get("jobPostings", []):
            external = job.get("externalPath", "")
            yield Job(
                source   = self.name,
                title    = job.get("title", ""),
                location = job.get("locationsText", "N/A"),
                url      = f"{base}{external}",
                job_id   = external or job.get("title", ""),
            )


class IndeedRSSSource(JobSource):
    """
    Fallback for companies on opaque ATS (iCIMS, Taleo, SuccessFactors).
    Subscribes to an Indeed search RSS feed scoped by query + location.
    Less precise than a direct API but covers what nothing else can.
    """
    kind = "indeed_rss"
    URL  = "https://www.indeed.com/rss"

    def __init__(
        self,
        name:     str,
        query:    str,
        location: str = "",
        filter_:  JobFilter | None = None,
    ) -> None:
        super().__init__(name, filter_)
        self.query    = query
        self.location = location

    @retry()
    def fetch_raw(self) -> ET.Element:
        r = HTTP.get(self.URL, params={"q": self.query, "l": self.location}, timeout=10)
        r.raise_for_status()
        return ET.fromstring(r.content)

    def parse_jobs(self, raw: ET.Element) -> Iterator[Job]:
        for item in raw.iter("item"):
            title_full = (item.findtext("title") or "").strip()
            url        = (item.findtext("link") or "").strip()
            guid       = (item.findtext("guid") or url).strip()
            # Indeed RSS titles are "Job Title - Company - Location"
            parts = [p.strip() for p in title_full.rsplit(" - ", 2)]
            if len(parts) == 3:
                title, _company, location = parts
            else:
                title, location = title_full, "N/A"
            yield Job(
                source=self.name, title=title, location=location, url=url, job_id=guid,
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
    sources:   list[JobSource]
    cache:     JobCache
    notifiers: list[Notifier]
    workers:   int = 8

    def run(self) -> list[Job]:
        log.info(f"polling {len(self.sources)} sources with {self.workers} workers")
        all_jobs = self._fetch_concurrent()

        new_jobs = [j for j in all_jobs if self.cache.is_new(j)]
        for job in new_jobs:
            self.cache.mark_seen(job)
        self.cache.save()

        log.info(f"{len(new_jobs)} new of {len(all_jobs)} total matching listings")
        for notifier in self.notifiers:
            notifier.notify(new_jobs)
        return new_jobs

    def _fetch_concurrent(self) -> list[Job]:
        all_jobs: list[Job] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(src.fetch): src for src in self.sources}
            for future in as_completed(futures):
                all_jobs.extend(future.result())
        return all_jobs


# ─── CONFIG ──────────────────────────────────────────────────────────────────

CACHE_PATH = Path.home() / ".job_monitor_cache.json"

EMAIL = EmailNotifier(
    sender    = "your_email@gmail.com",
    password  = "xxxx xxxx xxxx xxxx",  # Gmail App Password
    recipient = "your_email@gmail.com",
)

# Shared filter applied to every source unless overridden per-source
DEFAULT_FILTER = JobFilter(
    keywords = (
        "bioinformatics", "computational biolog", "genomics", "epigenomics",
        "sequencing", "oncology", "cancer", "data scientist", "machine learning",
        "methylation", "pipeline", "scientist", "bioinformatician",
    ),
)

# Per-source overrides built with `replace()` — preserves DEFAULT_FILTER's keywords
REMOTE_ONLY      = replace(DEFAULT_FILTER, remote_only=True)
CALIFORNIA_ONLY  = replace(DEFAULT_FILTER, location_contains="california")

# Adding a new company is one line: instantiate the appropriate Source.
SOURCES: list[JobSource] = [
    # Greenhouse
    GreenhouseSource("Natera",          "natera",         REMOTE_ONLY),
    GreenhouseSource("Guardant Health", "guardanthealth", DEFAULT_FILTER),
    GreenhouseSource("Grail",           "grail",          DEFAULT_FILTER),
    GreenhouseSource("Myriad Genetics", "myriadgenetics", DEFAULT_FILTER),
    GreenhouseSource("Kite Pharma",     "kitepharma",     CALIFORNIA_ONLY),
    GreenhouseSource("Personalis",      "personalis",     DEFAULT_FILTER),

    # Workday — tenant/board discovered from each company's public job board URL
    WorkdaySource("Gilead", tenant="gilead", board="gileadcareers", search="bioinformatics", filter_=DEFAULT_FILTER),
    WorkdaySource("Amgen",  tenant="amgen",  board="Amgen_Careers", search="bioinformatics", filter_=DEFAULT_FILTER),

    # Indeed RSS fallback for opaque ATS
    IndeedRSSSource("Indeed (Abbott Sylmar)",     query="bioinformatics Abbott", location="Sylmar, CA",     filter_=DEFAULT_FILTER),
    IndeedRSSSource("Indeed (Thermo West Hills)", query="bioinformatics Thermo", location="West Hills, CA", filter_=DEFAULT_FILTER),
]


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main() -> None:
    monitor = JobMonitor(
        sources   = SOURCES,
        cache     = JobCache(CACHE_PATH),
        notifiers = [ConsoleNotifier(), EMAIL],
    )
    monitor.run()


if __name__ == "__main__":
    main()