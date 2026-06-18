from celery.schedules import crontab
from ingestion.tasks import celery_app

# Define Celery Beat Schedules
# Timezone is configured to 'Africa/Lagos' in tasks.py, meaning these crontab hour specs match WAT.
celery_app.conf.beat_schedule = {
    "weekly-full-institutions-scrape": {
        "task": "ingestion.tasks.scrape_all_institutions",
        "schedule": crontab(hour=2, minute=0, day_of_week="monday"),
        "options": {"expires": 3600},
    },
    "daily-cbn-regulatory-scrape": {
        "task": "ingestion.tasks.scrape_cbn_regulatory",
        "schedule": crontab(hour=6, minute=0),
        "options": {"expires": 1800},
    },
    "periodic-institutions-scrape-30min": {
        "task": "ingestion.tasks.scrape_all_institutions",
        "schedule": 1800.0,  # 30 minutes in seconds
        "options": {"expires": 900},
    },
}
