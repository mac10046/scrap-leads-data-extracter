# Job Crawler Service


A FastAPI service that schedules Playwright-based crawling jobs to extract emails, phone numbers, social links, and contact pages. It supports:


- creating jobs from a list of sites (`POST /jobs`),
- uploading a CSV and enriching it with extracted contact details (`POST /jobs/csv`),
- polling job status (`GET /jobs/{job_id}`),
- downloading results as JSON or CSV (`GET /result/json/{job_id}`, `GET /result/csv/{job_id}`).


By default the server runs on port **5599**.


---


## Features


- Renders JS-heavy pages using Playwright.
- Normalizes phone numbers with `phonenumbers` (E.164 when possible).
- Respects `robots.txt` (best-effort).
- Job queue API: schedule jobs, check status, fetch results.
- CSV upload job: accepts CSV with a website column, crawls each row's site, and returns an enriched CSV with extracted contact info.


---


## Files included


- `job_crawler_service.py` — main FastAPI server and job runner. (Make sure you are using the patched version that contains `/jobs/csv`.)
- `requirements.txt` — Python dependencies.
- `email_extractor_postman_collection.json` — Postman collection (v2.1) with sample requests.


---


## Setup


1. Create & activate a virtual environment (recommended):


```bash
python -m venv venv
# mac / linux
source venv/bin/activate
# windows (PowerShell)
venv\Scripts\Activate.ps1
```


2. Install dependencies:


```bash
pip install -r requirements.txt
python -m playwright install
```


> **Important:** `python -m playwright install` downloads browser binaries required by Playwright.


---


## Run the service


Start the server (default port 5599):


```bash
python job_crawler_service.py
# or
uvicorn job_crawler_service:app --host 0.0.0.0 --port 5599 --log-level info
```


Open `http://localhost:5599/docs` for the automatic Swagger UI.


---


## API reference (quick)
MIT