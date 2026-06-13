"""Analytics computation service.

Generates daily, weekly, and monthly analytics reports from incident
data including trend analysis, waste type breakdowns, and camera statistics.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from civicsense.core.logging import get_logger
from civicsense.database.repositories.crud import (
    AnalyticsRepository,
    IncidentRepository,
)
from civicsense.events.event_bus import Event, EventType, get_event_bus

logger = get_logger("app")


class AnalyticsService:
    """Computes and caches analytics reports from incident data.

    Provides daily, weekly, and monthly analytics with
    trend analysis and breakdown statistics.
    """

    def __init__(self) -> None:
        """Initialize the AnalyticsService with repositories."""
        self._incident_repo = IncidentRepository()
        self._analytics_repo = AnalyticsRepository()
        self._event_bus = get_event_bus()

    def compute_daily(self, date_str: str | None = None) -> dict[str, Any]:
        """Compute analytics for a given day.

        Args:
            date_str: Date string in YYYY-MM-DD format. Defaults to today.

        Returns:
            Dictionary of daily analytics metrics.
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        date = datetime.strptime(date_str, "%Y-%m-%d")
        start = date.replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=1)

        incidents = self._incident_repo.get_by_date_range(start, end)
        return self._build_analytics(incidents, "daily", date)

    def compute_weekly(self, start_date: str | None = None) -> dict[str, Any]:
        """Compute analytics for a week.

        Args:
            start_date: Start date in YYYY-MM-DD format. Defaults to current week.

        Returns:
            Dictionary of weekly analytics metrics.
        """
        if start_date is None:
            today = datetime.now()
            start = today - timedelta(days=today.weekday())
            start_date = start.strftime("%Y-%m-%d")

        date = datetime.strptime(start_date, "%Y-%m-%d")
        end = date + timedelta(days=7)

        incidents = self._incident_repo.get_by_date_range(date, end)
        return self._build_analytics(incidents, "weekly", date)

    def compute_monthly(
        self, year: int | None = None, month: int | None = None
    ) -> dict[str, Any]:
        """Compute analytics for a month.

        Args:
            year: Calendar year. Defaults to current year.
            month: Calendar month (1-12). Defaults to current month.

        Returns:
            Dictionary of monthly analytics metrics.
        """
        now = datetime.now()
        year = year or now.year
        month = month or now.month

        start = datetime(year, month, 1)
        end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

        incidents = self._incident_repo.get_by_date_range(start, end)
        return self._build_analytics(incidents, "monthly", start)

    def get_trend_data(self, days: int = 30) -> list[dict[str, Any]]:
        """Get daily incident counts for trend analysis.

        Args:
            days: Number of past days to include.

        Returns:
            List of daily count dictionaries.
        """
        end = datetime.now()
        start = end - timedelta(days=days)
        incidents = self._incident_repo.get_by_date_range(start, end)

        daily_counts: dict[str, int] = {}
        for inc in incidents:
            day = inc.timestamp.strftime("%Y-%m-%d")
            daily_counts[day] = daily_counts.get(day, 0) + 1

        return [
            {"date": day, "count": count} for day, count in sorted(daily_counts.items())
        ]

    def get_camera_statistics(self) -> dict[str, dict[str, int]]:
        """Get per-camera incident statistics.

        Returns:
            Dictionary mapping camera IDs to their incident counts by status.
        """
        records = self._incident_repo.get_all(offset=0, limit=10000)
        stats: dict[str, dict[str, int]] = {}

        for record in records:
            cam = record.camera_id
            if cam not in stats:
                stats[cam] = {"total": 0, "pending": 0, "approved": 0, "rejected": 0}
            stats[cam]["total"] += 1
            if record.status in stats[cam]:
                stats[cam][record.status] += 1

        return stats

    def get_waste_type_statistics(self) -> dict[str, int]:
        """Get incident counts by waste type.

        Returns:
            Dictionary mapping waste types to their counts.
        """
        records = self._incident_repo.get_all(offset=0, limit=10000)
        stats: dict[str, int] = {}
        for record in records:
            stats[record.waste_type] = stats.get(record.waste_type, 0) + 1
        return stats

    def _build_analytics(
        self,
        incidents: list[Any],
        period: str,
        snapshot_date: datetime,
    ) -> dict[str, Any]:
        """Build an analytics report from a list of incidents.

        Args:
            incidents: List of incident records.
            period: The analytics period type.
            snapshot_date: The reference date for the report.

        Returns:
            Complete analytics report dictionary.
        """
        total = len(incidents)
        approved = sum(1 for i in incidents if i.status == "approved")
        rejected = sum(1 for i in incidents if i.status == "rejected")
        pending = sum(1 for i in incidents if i.status == "pending")

        waste_breakdown: dict[str, int] = {}
        hourly_dist: dict[int, int] = {h: 0 for h in range(24)}
        cameras: set[str] = set()
        confidences: list[float] = []

        for inc in incidents:
            waste_breakdown[inc.waste_type] = waste_breakdown.get(inc.waste_type, 0) + 1
            hourly_dist[inc.timestamp.hour] += 1
            cameras.add(inc.camera_id)
            confidences.append(inc.confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        report: dict[str, Any] = {
            "period": period,
            "date": snapshot_date.isoformat(),
            "total_incidents": total,
            "approved": approved,
            "rejected": rejected,
            "pending": pending,
            "unique_cameras": len(cameras),
            "waste_type_breakdown": waste_breakdown,
            "hourly_distribution": hourly_dist,
            "average_confidence": round(avg_confidence, 3),
        }

        self._analytics_repo.create(
            {
                "snapshot_date": snapshot_date,
                "period": period,
                "total_incidents": total,
                "approved_incidents": approved,
                "rejected_incidents": rejected,
                "pending_incidents": pending,
                "unique_cameras": len(cameras),
                "waste_type_breakdown": json.dumps(waste_breakdown),
                "hourly_distribution": json.dumps(hourly_dist),
                "average_confidence": avg_confidence,
            }
        )

        self._event_bus.publish(
            Event(
                event_type=EventType.ANALYTICS_READY,
                data=report,
                source="AnalyticsService",
            )
        )
        logger.info(
            f"Analytics computed: period={period}, incidents={total}",
            module="app",
        )
        return report
