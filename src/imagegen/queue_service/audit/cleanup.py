"""
Daemon thread that purges audit folders older than `retention_days`.

Walks data_predict/<service>/<Y>/<M>/<D>/, removes any day folder whose date
is more than retention_days old. Empty month/year folders are pruned
afterwards. The rmq_errors/ subtree is cleaned with the same policy.
"""

from __future__ import annotations

import shutil
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from ..config import AuditConfig
from ..observability import logger


class AuditCleaner:
    def __init__(self, cfg: AuditConfig):
        self.cfg = cfg
        self.service_root = Path(cfg.root_dir) / cfg.service_name

    def run_forever(self) -> None:
        if not self.cfg.enabled:
            logger.info("[audit-cleaner] disabled")
            return
        # Lazy import avoids circular: messaging → audit → ... → messaging.
        from ..messaging.shutdown import is_shutting_down

        interval_s = max(60, self.cfg.cleanup_interval_hours * 3600)
        logger.info(
            f"[audit-cleaner] retention={self.cfg.retention_days}d "
            f"interval={self.cfg.cleanup_interval_hours}h "
            f"root={self.service_root}"
        )
        # Run once at start, then every interval.
        while not is_shutting_down():
            try:
                self._run_once()
            except Exception as e:
                logger.exception(f"[audit-cleaner] error: {e}")
            # Sleep in 1s slices so SIGTERM is responsive.
            for _ in range(interval_s):
                if is_shutting_down():
                    return
                time.sleep(1)

    # ───────────────────── internals ─────────────────────

    def _run_once(self) -> None:
        cutoff = date.today() - timedelta(days=self.cfg.retention_days)
        removed = 0
        if self.service_root.exists():
            removed += self._purge_dated_tree(self.service_root, cutoff)
        rmq_root = self.service_root / "rmq_errors"
        if rmq_root.exists():
            removed += self._purge_dated_tree(rmq_root, cutoff)
        if removed:
            logger.info(
                f"[audit-cleaner] removed {removed} day-folder(s) older than "
                f"{cutoff}"
            )

    def _purge_dated_tree(self, root: Path, cutoff: date) -> int:
        """Remove day folders < cutoff. Return count removed."""
        removed = 0
        for year_dir in self._numeric_subdirs(root):
            for month_dir in self._numeric_subdirs(year_dir):
                for day_dir in self._numeric_subdirs(month_dir):
                    folder_date = self._folder_date(
                        year_dir.name, month_dir.name, day_dir.name
                    )
                    if folder_date is None or folder_date >= cutoff:
                        continue
                    try:
                        shutil.rmtree(day_dir)
                        removed += 1
                    except Exception as e:
                        logger.warning(
                            f"[audit-cleaner] rmtree {day_dir}: {e}"
                        )
                self._remove_if_empty(month_dir)
            self._remove_if_empty(year_dir)
        return removed

    @staticmethod
    def _numeric_subdirs(parent: Path):
        for p in parent.iterdir():
            if p.is_dir() and p.name.isdigit():
                yield p

    @staticmethod
    def _folder_date(y: str, m: str, d: str) -> date | None:
        try:
            return datetime(int(y), int(m), int(d)).date()
        except ValueError:
            return None

    @staticmethod
    def _remove_if_empty(d: Path) -> None:
        try:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass


def start_cleanup_thread(cfg: AuditConfig) -> None:
    """Spawn the audit cleaner daemon thread."""
    import threading

    cleaner = AuditCleaner(cfg)
    t = threading.Thread(
        target=cleaner.run_forever, name="audit-cleaner", daemon=True
    )
    t.start()
