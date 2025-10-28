#!/usr/bin/env python3
"""
job_crawler_service.py

Run:
    python job_crawler_service.py
or:
    uvicorn job_crawler_service:app --host 0.0.0.0 --port 5599

This script will start the FastAPI app on port 5599 by default when run directly.
"""

import asyncio
import json
import os
import re
import shutil
import sys
import time
import uuid
from collections import defaultdict
from typing import List, Optional

import tldextract
from bs4 import BeautifulSoup
import phonenumbers
import regex as re2

from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser

# Playwright imports
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from playwright._impl._errors import TargetClosedError

# ---------- Constants ----------
STORAGE_DIR = os.path.abspath("./jobs_storage")
os.makedirs(STORAGE_DIR, exist_ok=True)
RESOURCE_EXTS = ('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.svg', '.zip', '.rar', '.mp4', '.webm', '.ogg', '.mp3')

# ---------- Regex / Social hosts ----------
EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', re.I)
PHONE_RE = re2.compile(r'(\+?\d{1,3}[\s\-.(]*)?(?:\d[\s\-().]*){5,20}\d', re2.I)

SOCIAL_HOSTS = {
    'facebook': ('facebook.com', 'fb.me'),
    'instagram': ('instagram.com',),
    'twitter': ('twitter.com', 'x.com'),
    'linkedin': ('linkedin.com',),
    'youtube': ('youtube.com', 'youtu.be'),
    'telegram': ('t.me', 'telegram.me'),
    'github': ('github.com',),
    'whatsapp': ('wa.me', 'api.whatsapp.com', 'chat.whatsapp.com'),
    'tiktok': ('tiktok.com',),
}

# ---------- FastAPI app ----------
app = FastAPI(title="Email/Contact Crawler Job Service")

# In-memory job store (demo). For persistence, use DB.
JOBS = {}  # job_id -> metadata

# ---------- Pydantic models ----------
class JobRequest(BaseModel):
    sites: List[str] = Field(..., description="List of site root URLs to crawl (same-origin per site)")
    max_pages: int = Field(200, description="Max pages per site")
    concurrency: int = Field(4, description="Playwright worker concurrency per site")
    delay: float = Field(0.5, description="Delay between pages per worker (seconds)")
    headless: bool = Field(True, description="Run browser headless")
    default_region: Optional[str] = Field(None, description="Default region for phone parsing (e.g. 'IN', 'US')")
    job_name: Optional[str] = Field(None, description="Optional job name/label")

# ---------- Helper functions ----------
def same_domain(url1: str, url2: str) -> bool:
    e1 = tldextract.extract(url1)
    e2 = tldextract.extract(url2)
    return (e1.domain == e2.domain and e1.suffix == e2.suffix)

def normalize_url(base: str, link: str) -> Optional[str]:
    if not link:
        return None
    link = link.strip()
    if link.startswith('javascript:') or link.startswith('#'):
        return None
    try:
        return urljoin(base, link)
    except Exception:
        return None

def find_social_from_url(url: str) -> Optional[str]:
    try:
        nl = urlparse(url).netloc.lower()
    except Exception:
        return None
    for platform, hosts in SOCIAL_HOSTS.items():
        for h in hosts:
            if h in nl:
                return platform
    return None

def normalize_phone(raw: str, default_region: Optional[str] = None) -> Optional[str]:
    if not raw:
        return None
    candidate = re.sub(r'\s+', ' ', raw).strip()
    try:
        if default_region:
            parsed = phonenumbers.parse(candidate, default_region)
        else:
            parsed = phonenumbers.parse(candidate, None)
        if phonenumbers.is_possible_number(parsed) and phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        else:
            digits = re.sub(r'\D', '', candidate)
            if 6 <= len(digits) <= 15:
                return digits
    except phonenumbers.NumberParseException:
        digits = re.sub(r'\D', '', candidate)
        if 6 <= len(digits) <= 15:
            return digits
    return None

def parse_jsonld(soup: BeautifulSoup):
    out = {'sameAs': set(), 'contactPoint': []}
    for tag in soup.find_all('script', type='application/ld+json'):
        try:
            raw = tag.string
            if not raw:
                continue
            data = json.loads(raw)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            sa = item.get('sameAs')
            if sa:
                if isinstance(sa, list):
                    out['sameAs'].update(sa)
                else:
                    out['sameAs'].add(sa)
            cp = item.get('contactPoint') or item.get('contactPoints') or item.get('contact')
            if cp:
                out['contactPoint'].append(cp)
    return out

def extract_contacts_from_html(html: str, base_url: str, default_region: Optional[str] = None):
    soup = BeautifulSoup(html, 'html.parser')
    emails = set()
    phones = set()
    socials = defaultdict(set)
    contact_pages = set()

    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if href.lower().startswith('mailto:'):
            addr = href.split(':', 1)[1].split('?')[0].strip()
            if EMAIL_RE.search(addr):
                emails.add(addr)
        elif href.lower().startswith('tel:'):
            tel = href.split(':', 1)[1].split('?')[0].strip()
            norm = normalize_phone(tel, default_region)
            if norm:
                phones.add(norm)
        else:
            full = normalize_url(base_url, href)
            if not full:
                continue
            platform = find_social_from_url(full)
            if platform:
                socials[platform].add(full)
            lowhref = href.lower()
            if any(k in lowhref for k in ('contact', 'contact-us', 'contactus', 'get-in-touch', 'support', 'help', 'kontakt')):
                contact_pages.add(full)

    text = soup.get_text(separator=' ', strip=True)
    for em in EMAIL_RE.findall(text):
        emails.add(em)
    for ph_match in PHONE_RE.findall(text):
        candidate = ph_match if isinstance(ph_match, str) else ''.join(ph_match)
        norm = normalize_phone(candidate, default_region)
        if norm:
            phones.add(norm)

    j = parse_jsonld(soup)
    for s in j['sameAs']:
        platform = find_social_from_url(s)
        if platform:
            socials[platform].add(s)
    for cp in j['contactPoint']:
        if isinstance(cp, dict):
            e = cp.get('email')
            p = cp.get('telephone') or cp.get('phone') or cp.get('contactType')
            if e and EMAIL_RE.search(e):
                emails.add(e)
            if p:
                norm = normalize_phone(p, default_region)
                if norm:
                    phones.add(norm)
        elif isinstance(cp, list):
            for piece in cp:
                if isinstance(piece, dict):
                    e = piece.get('email')
                    p = piece.get('telephone') or piece.get('phone')
                    if e and EMAIL_RE.search(e):
                        emails.add(e)
                    if p:
                        norm = normalize_phone(p, default_region)
                        if norm:
                            phones.add(norm)

    return {
        'emails': emails,
        'phones': phones,
        'socials': {k: list(v) for k, v in socials.items()},
        'contact_pages': list(contact_pages)
    }

# ---------- Async site crawler (same as earlier) ----------
class AsyncSiteCrawler:
    def __init__(self, start_url: str, max_pages: int = 200, concurrency: int = 4,
                 delay: float = 0.5, headless: bool = True, default_region: Optional[str] = None):
        if not start_url.startswith('http'):
            start_url = 'http://' + start_url
        self.start_url = start_url.rstrip('/')
        self.max_pages = max_pages
        self.concurrency = max(1, concurrency)
        self.delay = delay
        self.headless = headless
        self.default_region = default_region

        self.visited = set()
        self.to_visit = asyncio.Queue()
        self.to_visit.put_nowait(self.start_url)

        self.results = {
            'emails': set(),
            'phones': set(),
            'socials': defaultdict(set),
            'contact_pages': set(),
            'pages_crawled': 0
        }

        self.robot = robotparser.RobotFileParser()
        try:
            start_origin = urlparse(self.start_url).scheme + "://" + urlparse(self.start_url).netloc
            self.robot.set_url(urljoin(start_origin, '/robots.txt'))
            self.robot.read()
            self.robot_ok = True
        except Exception:
            self.robot_ok = False

    def allowed(self, url: str) -> bool:
        if not self.robot_ok:
            return True
        try:
            return self.robot.can_fetch('*', url)
        except Exception:
            return True

    async def worker(self, browser, sem: asyncio.Semaphore, worker_id: int):
        context = await browser.new_context(ignore_https_errors=True)
        page = await context.new_page()
        await page.set_viewport_size({"width": 1280, "height": 800})

        try:
            while True:
                if self.results['pages_crawled'] >= self.max_pages:
                    break
                try:
                    url = await asyncio.wait_for(self.to_visit.get(), timeout=3.0)
                except asyncio.TimeoutError:
                    break

                if url in self.visited:
                    self.to_visit.task_done()
                    continue
                if not same_domain(self.start_url, url):
                    self.visited.add(url)
                    self.to_visit.task_done()
                    continue
                if not self.allowed(url):
                    self.visited.add(url)
                    self.to_visit.task_done()
                    continue

                await sem.acquire()
                try:
                    nav_ok = False
                    max_retries = 2
                    for attempt in range(1, max_retries + 1):
                        try:
                            resp = await page.goto(url, wait_until='domcontentloaded', timeout=60000)
                            try:
                                await page.wait_for_load_state('networkidle', timeout=3000)
                            except PlaywrightTimeoutError:
                                pass
                            content = await page.content()
                            nav_ok = True
                            break
                        except (PlaywrightTimeoutError, TargetClosedError, asyncio.CancelledError) as e:
                            print(f"[{self.start_url}][W{worker_id}] nav attempt {attempt} failed for {url}: {type(e).__name__}")
                            try:
                                await page.close()
                            except Exception:
                                pass
                            try:
                                await context.close()
                            except Exception:
                                pass
                            context = await browser.new_context(ignore_https_errors=True)
                            page = await context.new_page()
                            await page.set_viewport_size({"width": 1280, "height": 800})
                            await asyncio.sleep(0.5 * attempt)
                        except Exception as e:
                            print(f"[{self.start_url}][W{worker_id}] unexpected nav error for {url}: {e}")
                            break

                    if not nav_ok:
                        self.visited.add(url)
                        self.to_visit.task_done()
                        continue

                    parsed = extract_contacts_from_html(content, url, default_region=self.default_region)
                    self.results['emails'].update(parsed['emails'])
                    self.results['phones'].update(parsed['phones'])
                    for k, v in parsed['socials'].items():
                        self.results['socials'][k].update(v)
                    self.results['contact_pages'].update(parsed['contact_pages'])

                    soup = BeautifulSoup(content, 'html.parser')
                    for a in soup.find_all('a', href=True):
                        href = a['href'].strip()
                        if href.lower().startswith('mailto:') or href.lower().startswith('tel:'):
                            continue
                        full = normalize_url(url, href)
                        if not full:
                            continue
                        if full in self.visited:
                            continue
                        if not same_domain(self.start_url, full):
                            continue
                        if any(full.lower().endswith(ext) for ext in RESOURCE_EXTS):
                            continue
                        if len(full) > 400:
                            continue
                        await self.to_visit.put(full)

                    self.visited.add(url)
                    self.results['pages_crawled'] += 1

                    try:
                        await asyncio.sleep(self.delay)
                    except asyncio.CancelledError:
                        raise

                    self.to_visit.task_done()
                finally:
                    sem.release()
        except asyncio.CancelledError:
            print(f"[{self.start_url}] worker cancelled, cleaning up...")
        finally:
            try:
                await page.close()
            except Exception:
                pass
            try:
                await context.close()
            except Exception:
                pass

    async def run(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            sem = asyncio.Semaphore(self.concurrency)
            tasks = [asyncio.create_task(self.worker(browser, sem, i + 1)) for i in range(self.concurrency)]

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            finally:
                try:
                    await browser.close()
                except Exception:
                    pass

        return {
            'emails': sorted(self.results['emails']),
            'phones': sorted(self.results['phones']),
            'socials': {k: sorted(list(v)) for k, v in self.results['socials'].items()},
            'contact_pages': sorted(self.results['contact_pages']),
            'pages_crawled': self.results['pages_crawled']
        }

# ---------- Job orchestration ----------
async def run_job_worker(job_id: str):
    meta = JOBS.get(job_id)
    if not meta:
        return
    JOBS[job_id]['status'] = 'running'
    JOBS[job_id]['started_at'] = time.time()
    params = meta['params']

    job_dir = os.path.join(STORAGE_DIR, job_id)
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    os.makedirs(job_dir, exist_ok=True)

    aggregated = {
        'emails': set(),
        'phones': set(),
        'socials': defaultdict(set),
        'contact_pages': set(),
        'pages_crawled_total': 0,
        'per_site': {}
    }

    try:
        for site in params['sites']:
            JOBS[job_id]['current_site'] = site
            crawler = AsyncSiteCrawler(
                start_url=site,
                max_pages=params.get('max_pages', 200),
                concurrency=params.get('concurrency', 4),
                delay=params.get('delay', 0.5),
                headless=params.get('headless', True),
                default_region=params.get('default_region', None)
            )
            try:
                site_out = await crawler.run()
            except Exception as e:
                site_out = {
                    'emails': [],
                    'phones': [],
                    'socials': {},
                    'contact_pages': [],
                    'pages_crawled': 0,
                    'error': str(e)
                }
            for e in site_out.get('emails', []):
                aggregated['emails'].add(e)
            for p in site_out.get('phones', []):
                aggregated['phones'].add(p)
            for platform, links in site_out.get('socials', {}).items():
                for l in links:
                    aggregated['socials'][platform].add(l)
            for cp in site_out.get('contact_pages', []):
                aggregated['contact_pages'].add(cp)
            aggregated['pages_crawled_total'] += site_out.get('pages_crawled', 0)
            aggregated['per_site'][site] = site_out

        final = {
            'emails': sorted(aggregated['emails']),
            'phones': sorted(aggregated['phones']),
            'socials': {k: sorted(list(v)) for k, v in aggregated['socials'].items()},
            'contact_pages': sorted(aggregated['contact_pages']),
            'pages_crawled_total': aggregated['pages_crawled_total'],
            'per_site': aggregated['per_site'],
            'job_id': job_id,
            'params': params,
            'started_at': JOBS[job_id].get('started_at'),
            'finished_at': time.time()
        }

        json_path = os.path.join(job_dir, 'result.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final, f, indent=2, ensure_ascii=False)

        csv_path = os.path.join(job_dir, 'result.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write('type,value,meta\n')
            for e in final.get('emails', []):
                f.write(f'email,"{e}",\n')
            for ph in final.get('phones', []):
                f.write(f'phone,"{ph}",\n')
            for platform, links in final.get('socials', {}).items():
                for link in links:
                    f.write(f'social,"{link}",{platform}\n')
            for cp in final.get('contact_pages', []):
                f.write(f'contact_page,"{cp}",\n')

        JOBS[job_id]['status'] = 'done'
        JOBS[job_id]['finished_at'] = time.time()
        JOBS[job_id]['result_files'] = {'json': json_path, 'csv': csv_path}
    except Exception as e:
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = str(e)
        JOBS[job_id]['finished_at'] = time.time()

# ---------- API endpoints ----------
@app.post("/jobs", status_code=202)
async def create_job(req: JobRequest):
    sites = []
    for s in req.sites:
        s = s.strip()
        if not s:
            continue
        if not s.startswith('http'):
            s = 'https://' + s
        sites.append(s.rstrip('/'))
    if not sites:
        raise HTTPException(status_code=400, detail="No valid sites provided")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        'status': 'pending',
        'created_at': time.time(),
        'params': {
            'sites': sites,
            'max_pages': req.max_pages,
            'concurrency': req.concurrency,
            'delay': req.delay,
            'headless': req.headless,
            'default_region': req.default_region,
            'job_name': req.job_name
        },
        'result_files': None
    }

    asyncio.create_task(run_job_worker(job_id))
    return {"job_id": job_id, "status": "scheduled"}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str = Path(...)):
    meta = JOBS.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Job not found")
    view = {
        'job_id': job_id,
        'status': meta.get('status'),
        'created_at': meta.get('created_at'),
        'started_at': meta.get('started_at'),
        'finished_at': meta.get('finished_at'),
        'params': meta.get('params'),
        'current_site': meta.get('current_site'),
        'error': meta.get('error'),
        'result_files': meta.get('result_files')
    }
    return JSONResponse(content=view)

@app.get("/result/json/{job_id}")
async def get_result_json(job_id: str = Path(...)):
    meta = JOBS.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Job not found")
    files = meta.get('result_files')
    if not files or not files.get('json') or not os.path.exists(files.get('json')):
        raise HTTPException(status_code=404, detail="Result not ready")
    return FileResponse(files.get('json'), media_type='application/json', filename=f'{job_id}.json')

@app.get("/result/csv/{job_id}")
async def get_result_csv(job_id: str = Path(...)):
    meta = JOBS.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Job not found")
    files = meta.get('result_files')
    if not files or not files.get('csv') or not os.path.exists(files.get('csv')):
        raise HTTPException(status_code=404, detail="Result not ready")
    return FileResponse(files.get('csv'), media_type='text/csv', filename=f'{job_id}.csv')

@app.get("/health")
async def health():
    return {"status": "ok", "jobs_count": len(JOBS)}

# ---------- graceful shutdown helper ----------
def _shutdown_save():
    try:
        with open(os.path.join(STORAGE_DIR, 'jobs_summary.json'), 'w', encoding='utf-8') as f:
            json.dump({k: {**v, **{'params': v.get('params')}} for k, v in JOBS.items()}, f, indent=2, default=str)
    except Exception:
        pass

import signal
def _on_shutdown(sig, frame):
    print("Shutting down, saving job summary...")
    _shutdown_save()
    sys.exit(0)

signal.signal(signal.SIGINT, _on_shutdown)
signal.signal(signal.SIGTERM, _on_shutdown)

# ---------- Entrypoint: start uvicorn when run directly ----------
if __name__ == "__main__":
    # prefer uvicorn CLI but fall back to programmatic start for convenience
    try:
        import uvicorn
    except Exception:
        print("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)

    print("Starting job_crawler_service on port 5599 ...")
    # Adjust log_level as needed
    uvicorn.run(app, host="0.0.0.0", port=5599, log_level="info")
