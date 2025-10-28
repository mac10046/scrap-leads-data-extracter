# Email & Contact Extractor


A small FastAPI service that schedules Playwright-based site-crawling jobs to extract emails, phone numbers, social links, and contact pages. Runs on port **5599** by default.


## Files included
- `job_crawler_service.py` — main service (server + job runner)
- `requirements.txt` — Python dependencies
- `email_extractor_postman_collection.json` — Postman collection for testing the API


## Setup
1. Create a virtual environment (recommended):


```bash
python -m venv venv
source venv/bin/activate # on Windows: venv\Scripts\activate
```


2. Install dependencies:


```bash
pip install -r requirements.txt
python -m playwright install
```


> NOTE: Installing Playwright downloads browser binaries. This step is required for crawling JS-heavy sites.


## Run the service


Start the server (default port 5599):


```bash
python job_crawler_service.py
```


or using uvicorn directly:


```bash
uvicorn job_crawler_service:app --host 0.0.0.0 --port 5599 --log-level info
```


## API endpoints


- `POST /jobs` — create a job. JSON body (example):


```json
{
"sites": ["https://example.com"],
"max_pages": 50,
"concurrency": 2,
"delay": 1.0,
"headless": false,
"default_region": "IN",
"job_name": "batch-1"
}
```


Response: `{ "job_id": "<uuid>", "status": "scheduled" }`


- `GET /jobs/{job_id}` — get job status and metadata.
- `GET /result/json/{job_id}` — download aggregated result JSON (when job `status` is `done`).
- `GET /result/csv/{job_id}` — download CSV.
- `GET /health` — basic health check.


## Using Postman
1. Import `email_extractor_postman_collection.json` (Import > File) into Postman.
2. Use the `Create Job` request to schedule a job.
3. Use the returned `job_id` to poll `Get Job Status` and to fetch results.


## Notes & Caveats
- This demo stores jobs in memory and writes results to `./jobs_storage/<job_id>/`.
- If the service restarts, in-memory job state is lost (files remain on disk).
- For production, persist job metadata in a database (e.g. Redis, Postgres) and use a durable worker queue.
- Playwright and multiple concurrent jobs are resource intensive. Tune `concurrency` and `delay`.
- Respect `robots.txt` and site terms of service. Do not use this service to scrape sites you don't have permission to crawl.


## Troubleshooting
- If `python job_crawler_service.py` exits immediately, make sure `uvicorn` is installed. You can also run with `uvicorn` directly as above.
- If Playwright fails during a job, check that browsers are installed (`python -m playwright install`) and run a quick single-site test with `concurrency: 1` and `max_pages: 1`.


## License
MIT

---

## ☕ Support / Donate

This theme cost me **₹16,000 INR** to get designed, refined, and polished.  
If you find it useful (or just want to support an indie creator grinding in Mumbai):

❤️ Buy Me a Coffee → https://www.buymeacoffee.com/abdeali.c

Every coffee keeps the creativity brewing.

---

## 🤝 Contributing

Got ideas? Found a bug? Want to add a new feature?  
Pull Requests are open and welcome!

Fork → Improve → Contribute → High-five! 🙌

---

## 📬 Feedback & Contact

For customizations or full website development for financial firms:
🌍 https://softlancersolutions.com

---

### ⭐ Pro Tip

If you like this theme, **star the repository** on GitHub so it reaches more awesome devs like you 🌟

Let’s make the finance world look fabulous!
