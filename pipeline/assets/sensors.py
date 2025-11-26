from __future__ import annotations

from dagster import AssetSelection, RunRequest, SkipReason, define_asset_job, sensor

from pipeline.config import PATHS


materialize_all_assets_job = define_asset_job(
    "materialize_all_assets",
    selection=AssetSelection.all(),
)


def _parse_cursor(cursor: str | None) -> float:
    try:
        return float(cursor) if cursor is not None else 0.0
    except ValueError:
        return 0.0


@sensor(job=materialize_all_assets_job, minimum_interval_seconds=60)
def raw_data_sensor(context):
    """Trigger a full asset materialization when new raw data files appear."""

    data_dir = PATHS.raw_data
    if not data_dir.exists():
        reason = f"Raw data directory does not exist: {data_dir}"
        context.log.warning(reason)
        return SkipReason(reason)

    last_seen_ts = _parse_cursor(context.cursor)
    files = [path for path in data_dir.iterdir() if path.is_file()]
    if not files:
        return SkipReason("Raw data directory is empty")

    new_files: list[str] = []
    newest_ts = last_seen_ts
    for path in files:
        mtime = path.stat().st_mtime
        if mtime > last_seen_ts:
            new_files.append(path.name)
        if mtime > newest_ts:
            newest_ts = mtime

    if not new_files:
        return SkipReason("No new raw data files detected")

    context.update_cursor(str(newest_ts))
    run_key = f"raw-data-{int(newest_ts * 1000)}"

    return RunRequest(
        run_key=run_key,
        tags={
            "raw_data_dir": str(data_dir),
            "new_raw_files": ",".join(sorted(new_files)),
        },
    )
