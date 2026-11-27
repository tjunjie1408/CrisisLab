"""FDIC BankFind Suite API client with deterministic local snapshot caching."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

FDIC_BASE_URL = "https://api.fdic.gov/banks"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class FDICClient:
    """Fetches and caches raw FDIC BankFind API datasets as Parquet + SHA256 manifests."""

    def __init__(self, raw_dir: str | Path = "data/raw", timeout: float = 45.0) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _get_paginated(
        self,
        endpoint: str,
        *,
        filters: str | None = None,
        fields: list[str] | None = None,
        sort_by: str = "CERT",
        limit: int = 10000,
        single_page: bool = False,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        with httpx.Client(
            base_url=FDIC_BASE_URL,
            headers={"Accept": "application/json", "User-Agent": "crisislab/0.1"},
            timeout=self.timeout,
        ) as client:
            while True:
                params: dict[str, Any] = {
                    "limit": limit,
                    "offset": offset,
                    "sort_by": sort_by,
                    "sort_order": "ASC",
                }
                if filters:
                    params["filters"] = filters
                if fields:
                    params["fields"] = ",".join(fields)

                for attempt in range(4):
                    try:
                        resp = client.get(f"/{endpoint}", params=params)
                        resp.raise_for_status()
                    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                        if attempt == 3 or (
                            isinstance(exc, httpx.HTTPStatusError)
                            and exc.response.status_code not in {429, 500, 502, 503, 504}
                        ):
                            raise
                        time.sleep(2**attempt)
                    else:
                        break
                payload = resp.json()
                batch = [item["data"] for item in payload["data"]]
                rows.extend(batch)

                total = int(payload["meta"]["total"])
                offset += len(batch)
                if offset == total:
                    break
                if single_page or not batch or offset > total:
                    raise ValueError(
                        f"Inconsistent FDIC pagination for {endpoint}: {offset}/{total}"
                    )
        return rows

    def _record_manifest(self, name: str, path: Path, meta: dict[str, Any]) -> None:
        manifest_path = self.raw_dir / "manifest.json"
        existing: dict[str, Any] = {}
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing[name] = {
            "file": path.name,
            "sha256": _sha256_file(path),
            "fetched_at_utc": datetime.now(UTC).isoformat(),
            **meta,
        }
        manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def fetch_failures(self, *, refresh: bool = False) -> pd.DataFrame:
        out_path = self.raw_dir / "fdic_failures.parquet"
        if out_path.exists() and not refresh:
            return pd.read_parquet(out_path)

        fields = [
            "CERT",
            "NAME",
            "CITYST",
            "FAILDATE",
            "FAILYR",
            "RESTYPE",
            "RESTYPE1",
            "CHCLASS1",
            "QBFASSET",
            "QBFDEP",
            "COST",
        ]
        # ID is returned by the API but cannot be used as a sort key.
        rows = self._get_paginated("failures", fields=fields, sort_by="FAILDATE", single_page=True)
        rows.sort(key=lambda row: (row["FAILDATE"], row["ID"]))
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        self._record_manifest("failures", out_path, {"rows": len(df), "fields": fields})
        return df

    def fetch_institutions(self, *, refresh: bool = False) -> pd.DataFrame:
        out_path = self.raw_dir / "fdic_institutions.parquet"
        if out_path.exists() and not refresh:
            return pd.read_parquet(out_path)

        fields = [
            "CERT",
            "NAME",
            "STALP",
            "BKCLASS",
            "ACTIVE",
            "INACTIVE",
            "INSDATE",
            "ENDEFYMD",
            "EFFDATE",
            "PROCDATE",
            "CHANGEC1",
            "NEWCERT",
            "ULTCERT",
        ]
        rows = self._get_paginated("institutions", fields=fields, sort_by="CERT")
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        self._record_manifest("institutions", out_path, {"rows": len(df), "fields": fields})
        return df

    def fetch_financials_panel(
        self,
        *,
        start_year: int,
        end_year: int,
        quarters: list[str],
        camels_fields: list[str],
        refresh: bool = False,
    ) -> pd.DataFrame:
        out_path = self.raw_dir / f"fdic_financials_{start_year}_{end_year}.parquet"
        if out_path.exists() and not refresh:
            return pd.read_parquet(out_path)

        base_fields = ["CERT", "REPDTE", "NAME", "STALP", "BKCLASS"]
        fields = list(dict.fromkeys(base_fields + camels_fields))

        frames: list[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            for q in quarters:
                repdte = f"{year}{q}"
                rows = self._get_paginated(
                    "financials",
                    filters=f"REPDTE:{repdte}",
                    fields=fields,
                    sort_by="CERT",
                )
                if rows:
                    frames.append(pd.DataFrame(rows))
                print(f"[FDIC] {repdte}: {len(rows):,} bank-quarter rows", flush=True)

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=fields)
        df.to_parquet(out_path, index=False)
        self._record_manifest(
            f"financials_{start_year}_{end_year}",
            out_path,
            {
                "rows": len(df),
                "start_year": start_year,
                "end_year": end_year,
                "quarters": quarters,
                "fields": fields,
            },
        )
        return df
