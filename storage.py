"""Append submissions to a Google Sheet via a service account.

Uses the official Sheets API (through gspread) rather than the undocumented Google Forms
`formResponse` endpoint, because that endpoint returns 200 even when a submission is
silently dropped — unacceptable for data backing a paper.

Write volume is ~0.5/sec on average, comfortably under the ~60 writes/min per-service-account
quota, so rows are appended immediately. Bursts that hit the quota land in a bounded retry
queue and are flushed on the next successful call.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque

import gspread
from google.oauth2.service_account import Credentials

log = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_RETRY_QUEUE_MAX = 500

FIXED_LEADING = ["timestamp_utc", "session_uuid", "submission_seq", "mode", "app_version",
                 "sex", "age_band"]

# `urine_result` records whether the visitor had a urine test to answer the Proteinuria
# criterion from. Without it an untested visitor and one who tested negative are the same
# blank cell, and Proteinuria is worth a third of the score — see README.md.
#
# It is appended at the END rather than grouped with the other inputs on purpose:
# `_connect` only writes a header when the sheet is empty, so a sheet that already holds
# rows keeps its original column order. Inserting mid-header would silently misalign every
# subsequent row against the existing data; appending leaves the earlier columns intact.
#
# `eular_score` is a misnomer now that the weights are fitted rather than the published
# EULAR ones. Renaming it would misalign a live sheet the same way, so it stays.
FIXED_TRAILING = ["n_criteria", "eular_score", "band", "urine_result"]


def build_header(criteria: list[dict]) -> list[str]:
    """Build the sheet header row.

    Args:
        criteria: The criteria list from `core.load_criteria()`, in feature order.

    Returns:
        Column names: fixed metadata, then one column per criterion, then the result.
    """
    return FIXED_LEADING + [c["column"] for c in criteria] + FIXED_TRAILING


class SheetRecorder:
    """Appends rows to a worksheet, with a retry queue and a disabled (test) mode."""

    def __init__(self, header: list[str], enabled: bool = True):
        """Initialise the recorder without opening a connection.

        Args:
            header: Column names, from `build_header`.
            enabled: When False every append is a no-op that reports success. Used by test
                mode so demos and staff training never pollute the study dataset.
        """
        self._header = header
        self._enabled = enabled
        self._worksheet = None
        self._pending: deque[list] = deque(maxlen=_RETRY_QUEUE_MAX)

    @property
    def pending_count(self) -> int:
        """Number of rows waiting in the retry queue."""
        return len(self._pending)

    def _connect(self):
        """Open the worksheet and write the header if the sheet is empty."""
        if self._worksheet is not None:
            return self._worksheet
        raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        sheet_id = os.environ.get("SHEET_ID")
        if not raw or not sheet_id:
            raise RuntimeError(
                "GOOGLE_SERVICE_ACCOUNT_JSON and SHEET_ID must both be set. "
                "Create a service account, enable the Sheets API, and share the target "
                "sheet with the service account's email address."
            )
        creds = Credentials.from_service_account_info(json.loads(raw), scopes=_SCOPES)
        worksheet = gspread.authorize(creds).open_by_key(sheet_id).sheet1
        if not worksheet.acell("A1").value:
            worksheet.append_row(self._header, value_input_option="RAW")
        self._worksheet = worksheet
        return worksheet

    def _row_from(self, record: dict) -> list:
        return [record.get(col, "") for col in self._header]

    def append(self, record: dict) -> bool:
        """Append one submission, flushing any queued rows first.

        Args:
            record: Maps column name to value. Missing columns are written as empty.

        Returns:
            True if the row reached the sheet, False if it was queued for retry.
        """
        if not self._enabled:
            return True
        row = self._row_from(record)
        try:
            worksheet = self._connect()
            while self._pending:
                worksheet.append_row(self._pending[0], value_input_option="RAW")
                self._pending.popleft()
            worksheet.append_row(row, value_input_option="RAW")
            return True
        except Exception:
            log.exception("Sheets append failed; row queued for retry")
            if len(self._pending) == _RETRY_QUEUE_MAX:
                log.error("Retry queue full (%d rows) — dropping oldest", _RETRY_QUEUE_MAX)
            self._pending.append(row)
            self._worksheet = None
            return False
