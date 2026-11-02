"""
Job Manager for asynchronous and synchronous rice quality analysis.
"""

import asyncio
import io
import logging
import time
import uuid
from typing import Any, Dict, Optional

from ml.segmentation.pipeline import RiceQualityPipeline

from backend.app.services.run_logger import log_run, structured_logger

logger = logging.getLogger(__name__)

STAGES = [
    (1, "Validating image", 10),
    (2, "Detecting rice", 20),
    (3, "Segmenting grains", 35),
    (4, "Measuring grains", 50),
    (5, "Classifying defects", 65),
    (6, "Detecting foreign matter", 75),
    (7, "Computing sample statistics", 85),
    (8, "Checking standards", 95),
    (9, "Preparing report", 100),
]


class JobManager:
    """Manages analysis jobs and their lifecycle."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.pipeline = RiceQualityPipeline()

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress_stage": "Validating image",
            "progress_percent": 5,
            "created_at": time.time(),
            "completed_at": None,
            "error": None,
            "result": None,
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def process_sync(
        self,
        image_bytes: bytes,
        filename: str,
        manual_scale: Optional[Dict] = None,
        grade: str = "grade_a",
        profile: Optional[Any] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run analysis synchronously and store result."""
        job_id = self.create_job()
        job = self.jobs[job_id]
        job["status"] = "processing"

        t_start = time.monotonic()

        try:
            job["progress_stage"] = "Analyzing image through pipeline"
            job["progress_percent"] = 50
            result = self.pipeline.analyze(
                image_source=image_bytes,
                filename=filename,
                manual_scale=manual_scale,
                grade=grade,
                profile=profile,
                job_id=job_id,
                request_id=request_id,
            )
            job["status"] = "completed"
            job["progress_stage"] = "Completed"
            job["progress_percent"] = 100
            job["completed_at"] = time.time()
            job["result"] = result
            if isinstance(result, dict):
                result["job_id"] = job_id

            log_run(
                input_bytes=image_bytes,
                original_filename=filename,
                result=result,
                grade=grade,
            )

            return result
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            job["status"] = "failed"
            job["error"] = str(e)
            job["completed_at"] = time.time()

            structured_logger.error(
                "analysis_failed",
                job_id=job_id,
                status="failed",
                failed_stage="pipeline",
                error_type=type(e).__name__,
                error_message=str(e),
                total_processing_ms=int((time.monotonic() - t_start) * 1000),
            )

            error_result = {
                "success": False,
                "job_id": job_id,
                "error": str(e),
                "warnings": [str(e)],
            }
            log_run(
                input_bytes=image_bytes,
                original_filename=filename,
                result=error_result,
                grade=grade,
            )
            return error_result


# Global singleton
job_manager = JobManager()
