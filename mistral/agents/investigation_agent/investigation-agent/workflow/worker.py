"""Worker entrypoint: registers both investigation workflows and their activities."""

import asyncio

import mistralai.workflows as workflows

from investigation import InvestigationRouter, InvestigationThread

if __name__ == "__main__":
    asyncio.run(workflows.run_worker([InvestigationRouter, InvestigationThread]))
