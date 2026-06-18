import os
import sys
import time
import asyncio
import httpx

# Resolve import paths to find registry definitions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from registry.institutions import list_institutions

API_BASE_URL = os.environ.get("WAZOBIA_API_URL", "http://localhost:8000")
ADMIN_API_KEY = os.environ.get("WAZOBIA_ADMIN_KEY", "test-admin-secret-key-123")

def print_progress_table(results: list):
    """Formats and prints the standard ingestion execution progress table."""
    header = f"{'Institution':<14} | {'Status':<9} | {'Pages':<5} | {'Chunks':<6} | {'Points':<6} | {'Duration':<8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    # Sort results alphabetically by institution slug for consistent output
    for r in sorted(results, key=lambda x: x["institution"]):
        duration_str = f"{r['duration']}s" if isinstance(r['duration'], (int, float)) else r['duration']
        print(f"{r['institution']:<14} | {r['status']:<9} | {r['pages']:<5} | {r['chunks']:<6} | {r['points']:<6} | {duration_str:<8}")
    print("=" * len(header) + "\n")

async def run_ingestion_for_institution(client: httpx.AsyncClient, slug: str, headers: dict) -> dict:
    """Triggers and polls ingestion for a single institution asynchronously."""
    result = {
        "institution": slug.upper(),
        "status": "FAILED",
        "pages": "-",
        "chunks": "-",
        "points": "-",
        "duration": "-",
    }
    
    # 1. Trigger Ingestion
    trigger_url = f"{API_BASE_URL}/v1/institutions/{slug}/ingest"
    try:
        response = await client.post(trigger_url, headers=headers, timeout=30.0)
    except Exception as e:
        result["duration"] = f"error: {str(e)[:15]}"
        return result

    if response.status_code == 429:
        result["duration"] = "error: 429 rate limit"
        return result
    if response.status_code not in [200, 202]:
        result["duration"] = f"error: HTTP {response.status_code}"
        return result

    trigger_data = response.json()
    task_id = trigger_data.get("task_id")
    if not task_id:
        result["duration"] = "error: no task id"
        return result

    # 2. Poll Task Status (wait up to 5 minutes)
    status_url = f"{API_BASE_URL}/v1/ingest/tasks/{task_id}"
    start_time = time.time()
    timeout_limit = 300  # 5 minutes
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_limit:
            result["duration"] = "error: timeout"
            break
            
        try:
            status_resp = await client.get(status_url, headers=headers, timeout=10.0)
            if status_resp.status_code == 200:
                task_data = status_resp.json()
                status = task_data.get("status")
                
                # Check status
                if task_data.get("ready"):
                    if status == "SUCCESS" or (task_data.get("result") and "error" not in task_data["result"]):
                        task_res = task_data.get("result", {})
                        result["status"] = "SUCCESS"
                        result["pages"] = str(task_res.get("pages_scraped", 0))
                        result["chunks"] = str(task_res.get("chunks_created", 0))
                        result["points"] = str(task_res.get("points_upserted", 0))
                        result["duration"] = f"{int(task_res.get('duration_seconds', elapsed))}s"
                    else:
                        task_res = task_data.get("result") or {}
                        err_msg = task_res.get("error", "unknown error")
                        result["duration"] = f"error: {str(err_msg)[:15]}"
                    break
        except Exception:
            # Keep polling in case of transient connection error
            pass
            
        await asyncio.sleep(5)
        
    return result

async def worker(sem: asyncio.Semaphore, client: httpx.AsyncClient, inst, headers: dict, results: list):
    """Worker wrapper to execute ingestion under a concurrency-limiting semaphore."""
    async with sem:
        print(f"[{inst.slug.upper()}] Starting ingestion trigger...")
        res = await run_ingestion_for_institution(client, inst.slug, headers)
        results.append(res)
        print(f"[{inst.slug.upper()}] Ingestion finished. Status: {res['status']}. Duration: {res['duration']}")

async def main():
    print("Starting Wazobia Bulk Ingestion Pipeline (Max Concurrency: 5)...")
    active_institutions = list_institutions(active_only=True)
    if not active_institutions:
        print("No active institutions registered in registry.")
        sys.exit(0)

    headers = {"X-API-Key": ADMIN_API_KEY}
    results = []
    
    # Limit to 5 concurrent tasks
    sem = asyncio.Semaphore(5)

    async with httpx.AsyncClient() as client:
        tasks = [
            worker(sem, client, inst, headers, results)
            for inst in active_institutions
        ]
        await asyncio.gather(*tasks)

    print_progress_table(results)

    any_failed = any(r["status"] != "SUCCESS" for r in results)
    if any_failed:
        print("Bulk ingestion finished with errors.")
        sys.exit(1)
    else:
        print("Bulk ingestion completed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
