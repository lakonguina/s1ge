"""Worker module for handling scheduled tasks.

This module configures the Celery application and schedules various tasks
for maintaining the investment platform.
"""

import os
from celery import Celery

# Import tasks from their respective modules
from src.transactions import get_declarations

app = Celery('worker', broker=os.getenv("REDIS_URL"))

# Register Celery tasks
app.task(get_declarations)

# Configure Celery beat schedule
app.conf.beat_schedule = {
    'get-declarations': {
        'task': 'src.transactions.get_declarations',
        'schedule': 86400.0
    },
}