#!/usr/bin/env python3
"""
Classify compounds from InChIKeys, SMILES, or InChI strings.

The user selects one or more services with --services. When ClassyFire is
selected, FiehnLab is used automatically as a ChemOnt fallback for compounds
that ClassyFire cannot classify.

The input file must contain one compound identifier per line. Blank lines and
lines starting with '#' are ignored. Results are written as JSON to stdout by
default.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import requests

try:
    from rdkit import Chem
    from rdkit.Chem import inchi as rd_inchi
except ImportError:  # pragma: no cover - optional dependency
    Chem = None  # type: ignore[assignment]
    rd_inchi = None  # type: ignore[assignment]


DEFAULT_TIMEOUT = 30
INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
CLASSYFIRE_BASE_URL = "http://classyfire.wishartlab.com"
FIEHNLAB_ENTITY_URL = "https://cfb.fiehnlab.ucdavis.edu/entities/{inchikey}.json"
OLS_SEARCH_URL = "https://www.ebi.ac.uk/ols4/api/search"
DEFAULT_CHEBI_OBO = Path(__file__).with_name("chebi.obo")
DEFAULT_OUTPUT_FORMAT = "json"
DEFAULT_USER_AGENT = "msprops2-compound-classifier/0.1 (+https://github.com/)"
CLASSYFIRE_RATE_SECONDS = 12.0
CLASSYFIRE_BATCH_SIZE = 1000
CLASSYFIRE_POLL_INTERVAL_SECONDS = 15.0
CLASSYFIRE_MAX_WAIT_SECONDS = 60.0
FIEHNLAB_RATE_SECONDS = 0.5
CHEBI_RATE_SECONDS = 12.0
PUBCHEM_RATE_SECONDS = 0.2
PUBCHEM_MINUTE_CAP = 400
PUBCHEM_BATCH_SIZE = 10
MAX_RETRIES = 3


@dataclass
class ClassificationRecord:
    """Normalized classification information for a single InChIKey."""

    inchikey: str
    classification_name: Optional[str]
    source: str
    classification_id: Optional[str] = None
    classification_url: Optional[str] = None
    note: Optional[str] = None
    kingdom: Optional[str] = None
    superclass: Optional[str] = None
    class_level: Optional[str] = None
    subclass: Optional[str] = None
    other_classes: Optional[str] = None
    hierarchies: Optional[List[List[Dict[str, str]]]] = None


@dataclass
class NormalizedCompound:
    raw_input: str
    input_format: str
    inchikey: str
    smiles: Optional[str] = None
    inchi: Optional[str] = None


class RateLimiter:
    """Simple sliding window rate limiter shared across threads."""

    def __init__(self, max_calls: int, interval: float) -> None:
        if max_calls <= 0 or interval <= 0:
            raise ValueError("max_calls and interval must be > 0")
        self.max_calls = max_calls
        self.interval = interval
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until registering a new call keeps us within rate limits."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.interval:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                sleep_for = self.interval - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)


class CompositeRateLimiter:
    """Combine multiple rate limiters (e.g., short-term + long-term)."""

    def __init__(self, limiters: Sequence[RateLimiter]):
        if not limiters:
            raise ValueError("CompositeRateLimiter requires at least one limiter")
        self.limiters = list(limiters)

    def acquire(self) -> None:
        for limiter in self.limiters:
            limiter.acquire()


def chunked(sequence: Sequence[str], size: int) -> Iterator[List[str]]:
    """Yield lists of up to `size` items from `sequence`."""
    if size <= 0:
        raise ValueError("Chunk size must be > 0")
    for start in range(0, len(sequence), size):
        yield list(sequence[start : start + size])


def ensure_rdkit_available() -> None:
    if Chem is None or rd_inchi is None:
        raise RuntimeError(
            "RDKit is required for SMILES/InChI inputs. Install with 'pip install rdkit-pypi'."
        )


def is_inchikey(value: str) -> bool:
    return bool(INCHIKEY_RE.match(value))


def detect_input_format(value: str) -> str:
    if value.startswith("InChI="):
        return "inchi"
    if is_inchikey(value):
        return "inchikey"
    return "smiles"


def normalize_identifier(raw: str) -> NormalizedCompound:
    cleaned = raw.strip()
    fmt = detect_input_format(cleaned)
    if fmt == "inchikey":
        return NormalizedCompound(raw_input=cleaned, input_format=fmt, inchikey=cleaned.upper())

    ensure_rdkit_available()
    mol = None
    if fmt == "inchi":
        try:
            mol = rd_inchi.MolFromInchi(cleaned, sanitize=True)
        except Exception as exc:  # pragma: no cover - rdkit errors
            raise ValueError(f"Failed to parse InChI: {exc}") from exc
    else:  # smiles
        mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        raise ValueError(f"Unable to parse {fmt} input")
    try:
        smiles = Chem.MolToSmiles(mol)
        inchi = rd_inchi.MolToInchi(mol)
        inchikey = rd_inchi.MolToInchiKey(mol)
    except Exception as exc:  # pragma: no cover - rdkit errors
        raise ValueError(f"Failed to convert molecule: {exc}") from exc
    if not inchikey or not is_inchikey(inchikey):
        raise ValueError("Could not derive valid InChIKey from input")
    return NormalizedCompound(
        raw_input=cleaned,
        input_format=fmt,
        inchikey=inchikey,
        smiles=smiles,
        inchi=inchi,
    )


def format_taxon(node: Optional[Dict]) -> Optional[str]:
    if not node or not isinstance(node, dict):
        return None
    name = node.get("name")
    chem_id = node.get("chemont_id") or node.get("obo_id")
    if name and chem_id:
        return f"{name} ({chem_id})"
    return name or chem_id


def join_taxa(nodes: Iterable[Dict]) -> Optional[str]:
    formatted = [format_taxon(node) for node in nodes if format_taxon(node)]
    return "; ".join(formatted) if formatted else None


def prune_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, child in value.items():
            pruned = prune_nulls(child)
            if pruned is not None:
                cleaned[key] = pruned
        return cleaned or None
    if isinstance(value, list):
        cleaned_list: List[Any] = []
        for item in value:
            pruned = prune_nulls(item)
            if pruned is not None:
                cleaned_list.append(pruned)
        return cleaned_list or None
    return value


class ChebiOntology:
    def __init__(self, path: Path) -> None:
        self.terms: Dict[str, Dict[str, Any]] = {}
        self._load(path)

    def _commit(self, term: Optional[Dict[str, Any]]) -> None:
        if term and term.get("id"):
            term_id = term["id"]
            self.terms[term_id] = {
                "name": term.get("name"),
                "parents": term.get("parents", []),
            }

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"ChEBI ontology file not found: {path}")
        current: Optional[Dict[str, Any]] = None
        in_term = False
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line == "[Term]":
                    self._commit(current)
                    current = {"parents": []}
                    in_term = True
                    continue
                if not line:
                    self._commit(current)
                    current = None
                    in_term = False
                    continue
                if line.startswith("[") and line != "[Term]":
                    self._commit(current)
                    current = None
                    in_term = False
                    continue
                if not in_term or current is None:
                    continue
                if line.startswith("id: "):
                    current["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    current["name"] = line[6:].strip()
                elif line.startswith("is_a: "):
                    parent = line.split("is_a: ", 1)[1].split("!")[0].strip()
                    current.setdefault("parents", []).append(parent)
        self._commit(current)

    def hierarchies(self, term_id: str) -> List[List[Dict[str, str]]]:
        term = self.terms.get(term_id)
        if not term:
            return []
        parents = term.get("parents") or []
        paths: List[List[Dict[str, str]]] = []
        for parent_id in parents:
            path: List[Dict[str, str]] = []
            current = parent_id
            visited: set[str] = set()
            while current and current not in visited:
                visited.add(current)
                info = self.terms.get(current)
                if not info:
                    break
                path.append({
                    "id": current,
                    "name": info.get("name") or current,
                })
                parent_list = info.get("parents") or []
                current = parent_list[0] if parent_list else None
            if path:
                paths.append(path)
        return paths


class BaseClient:
    def __init__(
        self,
        session: requests.Session,
        rate_limiter: RateLimiter,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> None:
        self.session = session
        self.rate_limiter = rate_limiter
        self.max_retries = max(1, max_retries)
        self.backoff = max(0.1, backoff)

    def _request(
        self,
        method: str,
        url: str,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        retries: Optional[int] = None,
        allow_error: bool = False,
        **kwargs,
    ) -> Optional[requests.Response]:
        last_error: Optional[str] = None
        max_attempts = retries if retries is not None else self.max_retries
        for attempt in range(1, max_attempts + 1):
            limiter = rate_limiter or self.rate_limiter
            if limiter:
                limiter.acquire()
            try:
                response = self.session.request(method, url, timeout=DEFAULT_TIMEOUT, **kwargs)
            except requests.RequestException as exc:  # network/timeout etc.
                last_error = str(exc)
            else:
                if allow_error:
                    return response
                if response.status_code in (429, 503, 504):
                    # Retry with exponential backoff.
                    last_error = f"HTTP {response.status_code}"
                elif 200 <= response.status_code < 400:
                    return response
                else:
                    last_error = f"HTTP {response.status_code}"
                    break
            time.sleep(self.backoff * attempt)
        if last_error:
            sys.stderr.write(f"[warn] {self.__class__.__name__} request failed for {url}: {last_error}\n")
        return None


class ClassyFireClient(BaseClient):
    BASE_URL = CLASSYFIRE_BASE_URL
    QUERIES_ENDPOINT = f"{BASE_URL}/queries.json"
    STATUS_ENDPOINT = f"{BASE_URL}/queries/{{query_id}}/status.json"
    RESULTS_ENDPOINT = f"{BASE_URL}/queries/{{query_id}}.json"

    def __init__(
        self,
        session: requests.Session,
        rate_limiter: RateLimiter,
        *,
        batch_size: int = 1000,
        poll_interval: float = 15.0,
        max_wait: float = 600.0,
        per_page: int = 100,
        status_rate_limiter: Optional[RateLimiter] = None,
        max_status_stall_checks: int = 6,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> None:
        super().__init__(session, rate_limiter, max_retries=max_retries, backoff=backoff)
        self.batch_size = max(1, min(batch_size, 1000))
        self.poll_interval = max(1.0, poll_interval)
        self.max_wait = max(self.poll_interval, max_wait)
        self.per_page = max(1, min(per_page, 100))
        self.status_rate_limiter = status_rate_limiter or RateLimiter(max_calls=10, interval=60)
        self.max_status_stall_checks = max(1, max_status_stall_checks)
        self._status_failures = 0

    def bulk_fetch(
        self, inchikeys: Sequence[str]
    ) -> Tuple[Dict[str, ClassificationRecord], List[str], Dict[str, str]]:
        records: Dict[str, ClassificationRecord] = {}
        missing: List[str] = []
        errors: Dict[str, str] = {}

        if not inchikeys:
            return records, missing, errors

        normalized_input = [key.upper() for key in inchikeys]

        for chunk_index, chunk in enumerate(chunked(normalized_input, self.batch_size)):
            chunk_records, chunk_missing, chunk_errors = self._process_chunk(chunk_index, chunk)
            for key, record in chunk_records.items():
                records[key] = record
            missing.extend(chunk_missing)
            for key, message in chunk_errors.items():
                errors[key] = message
        return records, missing, errors

    def _process_chunk(
        self, chunk_index: int, chunk: Sequence[str]
    ) -> Tuple[Dict[str, ClassificationRecord], List[str], Dict[str, str]]:
        label = f"compound_classificator_{int(time.time())}_{chunk_index}"
        payload = {
            "label": label,
            "query_input": "\n".join(f"InChIKey={key}" for key in chunk),
            "query_type": "STRUCTURE",
        }
        response = self._request("POST", self.QUERIES_ENDPOINT, json=payload)
        if response is None:
            return {}, list(chunk), {key: "Request failed" for key in chunk}
        if response.status_code not in (200, 201):
            message = f"HTTP {response.status_code}"
            return {}, list(chunk), {key: message for key in chunk}
        try:
            submission = response.json()
        except json.JSONDecodeError:
            return {}, list(chunk), {key: "Invalid JSON response" for key in chunk}

        query_id = submission.get("id")
        if not isinstance(query_id, int):
            return {}, list(chunk), {key: "Missing query identifier" for key in chunk}

        completed, failure_reason = self._wait_for_completion(query_id)
        if not completed:
            message = failure_reason or "Classification did not complete"
            return {}, list(chunk), {key: message for key in chunk}

        try:
            entities, invalid_inputs = self._collect_results(query_id)
        except ValueError as exc:
            return {}, list(chunk), {key: str(exc) for key in chunk}

        entity_records = self._records_from_entities(entities)
        invalid_keys = {
            key
            for key in (self._normalize_invalid_entry(entry) for entry in invalid_inputs)
            if key is not None
        }

        missing: List[str] = []
        errors: Dict[str, str] = {}
        for key in chunk:
            record = entity_records.get(key)
            if record is None:
                if key in invalid_keys:
                    errors[key] = "Invalid input"
                else:
                    errors[key] = "Not found in ClassyFire response"
                missing.append(key)
                continue
            if record.classification_name is None:
                errors.setdefault(key, record.note or "No classification returned")
                missing.append(key)

        return entity_records, missing, errors

    def _wait_for_completion(self, query_id: int) -> Tuple[bool, Optional[str]]:
        deadline = time.monotonic() + self.max_wait
        last_status: Optional[str] = None
        missing_status_polls = 0
        next_check_at = time.monotonic() + self.poll_interval
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_check_at:
                time.sleep(next_check_at - now)
            status, status_code = self._check_status(query_id)
            if status_code == 429:
                self._status_failures += 1
                if self._status_failures >= 2:
                    return False, "Rate limited by ClassyFire status endpoint"
                next_check_at = time.monotonic() + self.poll_interval
                continue
            else:
                self._status_failures = 0
            normalized = (status or "").strip().lower()
            if normalized == "done":
                self._status_failures = 0
                return True, None
            if normalized in {"failed", "error"}:
                self._status_failures = 0
                return False, status
            if status is None:
                missing_status_polls += 1
            else:
                missing_status_polls = 0
            if status and status.isdigit():
                missing_status_polls = 0
            if missing_status_polls >= self.max_status_stall_checks:
                hint = last_status or "unknown"
                return False, f"No status received from ClassyFire (last status: {hint})"
            if status:
                last_status = status
            next_check_at = time.monotonic() + self.poll_interval
        self._status_failures = 0
        return False, "Timed out waiting for ClassyFire query"

    def _check_status(self, query_id: int) -> Tuple[Optional[str], Optional[int]]:
        url = self.STATUS_ENDPOINT.format(query_id=query_id)
        response = self._request(
            "GET",
            url,
            rate_limiter=self.status_rate_limiter,
            retries=1,
            allow_error=True,
        )
        if response is None:
            return None, None
        return response.text.strip(), response.status_code

    def _collect_results(self, query_id: int) -> Tuple[List[Dict], List[str]]:
        entities: List[Dict] = []
        invalid_inputs: List[str] = []
        page = 1
        total_pages = 1
        while page <= total_pages:
            params = {"page": page, "per_page": self.per_page}
            url = self.RESULTS_ENDPOINT.format(query_id=query_id)
            response = self._request("GET", url, params=params, rate_limiter=self.status_rate_limiter)
            if response is None:
                raise ValueError("Failed to retrieve query results")
            try:
                payload = response.json()
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON response for query results")

            entities.extend(payload.get("entities") or [])
            invalid_inputs.extend(payload.get("invalid_entities") or [])
            total_pages = max(total_pages, int(payload.get("number_of_pages") or 1))
            page += 1
        return entities, invalid_inputs

    def _records_from_entities(self, entities: Sequence[Dict]) -> Dict[str, ClassificationRecord]:
        records: Dict[str, ClassificationRecord] = {}
        for entity in entities:
            normalized_key = self._normalize_inchikey(entity.get("inchikey"))
            if not normalized_key:
                continue
            direct_parent = entity.get("direct_parent") or {}
            classification_name = direct_parent.get("name")
            classification_id = direct_parent.get("chemont_id")
            classification_url = direct_parent.get("url")
            note: Optional[str] = None
            if not classification_name:
                fallback = entity.get("class") or {}
                classification_name = fallback.get("name")
                classification_id = fallback.get("chemont_id")
                classification_url = fallback.get("url")
                if classification_name:
                    note = "Direct parent missing; class used instead"
                else:
                    note = "No direct parent returned"

            kingdom = format_taxon(entity.get("kingdom"))
            superclass = format_taxon(entity.get("superclass"))
            class_level = format_taxon(entity.get("class"))
            subclass = format_taxon(entity.get("subclass"))
            other_nodes: List[Dict] = []
            for key in ("alternative_parents", "ancestors", "intermediate_nodes", "substituents"):
                nodes = entity.get(key)
                if nodes:
                    if isinstance(nodes, dict):
                        other_nodes.append(nodes)
                    else:
                        other_nodes.extend(nodes)
            other_classes = join_taxa(other_nodes)
            records[normalized_key] = ClassificationRecord(
                inchikey=normalized_key,
                classification_name=classification_name,
                classification_id=classification_id,
                classification_url=classification_url,
                source="ClassyFire",
                note=note,
                kingdom=kingdom,
                superclass=superclass,
                class_level=class_level,
                subclass=subclass,
                other_classes=other_classes,
            )
        return records

    @staticmethod
    def _normalize_invalid_entry(entry: object) -> Optional[str]:
        candidate: Optional[str] = None
        if isinstance(entry, dict):
            candidate = (
                entry.get("input")
                or entry.get("query_input")
                or entry.get("structure")
                or entry.get("entity")
            )
        elif entry is not None:
            candidate = str(entry)
        return ClassyFireClient._normalize_inchikey(candidate)

    @staticmethod
    def _normalize_inchikey(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        value = value.strip()
        if not value:
            return None
        if "=" in value:
            _, tail = value.split("=", 1)
        else:
            tail = value
        normalized = tail.strip().upper()
        return normalized if normalized else None


class FiehnLabClient(BaseClient):
    BASE_URL = FIEHNLAB_ENTITY_URL

    def bulk_fetch(
        self, inchikeys: Sequence[str]
    ) -> Tuple[Dict[str, ClassificationRecord], List[str], Dict[str, str]]:
        records: Dict[str, ClassificationRecord] = {}
        missing: List[str] = []
        errors: Dict[str, str] = {}
        for key in inchikeys:
            record = self.fetch(key)
            if record.classification_name:
                records[key] = record
            else:
                missing.append(key)
                if record.note:
                    errors[key] = record.note
        return records, missing, errors

    def fetch(self, inchikey: str) -> ClassificationRecord:
        url = self.BASE_URL.format(inchikey=inchikey)
        response = self._request("GET", url)
        if response is None:
            return ClassificationRecord(
                inchikey=inchikey,
                classification_name=None,
                source="FiehnLab",
                note="Request failed",
            )
        if response.status_code == 404:
            return ClassificationRecord(
                inchikey=inchikey,
                classification_name=None,
                source="FiehnLab",
                note="Not found",
            )
        if response.status_code != 200:
            return ClassificationRecord(
                inchikey=inchikey,
                classification_name=None,
                source="FiehnLab",
                note=f"HTTP {response.status_code}",
            )
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return ClassificationRecord(
                inchikey=inchikey,
                classification_name=None,
                source="FiehnLab",
                note="Invalid JSON response",
            )
        return self._record_from_payload(payload, fallback_inchikey=inchikey)

    def _record_from_payload(self, payload: Dict, fallback_inchikey: str) -> ClassificationRecord:
        normalized = self._normalize_key(payload.get("inchikey")) or fallback_inchikey
        direct_parent = payload.get("direct_parent") or {}
        classification_name = direct_parent.get("name")
        classification_id = direct_parent.get("chemont_id")
        classification_url = direct_parent.get("url")
        note: Optional[str] = None
        if not classification_name:
            class_node = payload.get("class") or {}
            classification_name = class_node.get("name")
            classification_id = class_node.get("chemont_id")
            classification_url = class_node.get("url")
            if classification_name:
                note = "Direct parent missing; class used instead"
            else:
                note = "No classification returned"
        kingdom = format_taxon(payload.get("kingdom"))
        superclass = format_taxon(payload.get("superclass"))
        class_level = format_taxon(payload.get("class"))
        subclass = format_taxon(payload.get("subclass"))
        other_nodes: List[Dict] = []
        for key in ("alternative_parents", "ancestors", "intermediate_nodes", "substituents"):
            nodes = payload.get(key)
            if nodes:
                if isinstance(nodes, dict):
                    other_nodes.append(nodes)
                else:
                    other_nodes.extend(nodes)
        other_classes = join_taxa(other_nodes)
        return ClassificationRecord(
            inchikey=normalized,
            classification_name=classification_name,
            classification_id=classification_id,
            classification_url=classification_url,
            source="FiehnLab",
            note=note,
            kingdom=kingdom,
            superclass=superclass,
            class_level=class_level,
            subclass=subclass,
            other_classes=other_classes,
        )

    @staticmethod
    def _normalize_key(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        cleaned = value.split("=", 1)[-1].strip().upper()
        return cleaned if is_inchikey(cleaned) else None


class PubChemClient(BaseClient):
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/classification/JSON"

    def bulk_fetch(
        self, inchikeys: Sequence[str]
    ) -> Tuple[Dict[str, ClassificationRecord], List[str], Dict[str, str]]:
        if not inchikeys:
            return {}, [], {}
        data = {"inchikey": ",".join(inchikeys)}
        response = self._request("POST", self.BASE_URL, data=data)
        errors: Dict[str, str] = {}
        if response is None:
            for key in inchikeys:
                errors[key] = "Request failed"
            return {}, list(inchikeys), errors
        if response.status_code != 200:
            message = f"HTTP {response.status_code}"
            for key in inchikeys:
                errors[key] = message
            return {}, list(inchikeys), errors
        try:
            payload = response.json()
        except json.JSONDecodeError:
            for key in inchikeys:
                errors[key] = "Invalid JSON response"
            return {}, list(inchikeys), errors
        records, missing = self._parse_payload(payload, inchikeys)
        for key in missing:
            errors.setdefault(key, "No classification in response")
        return records, missing, errors

    @staticmethod
    def _parse_payload(payload: Dict, requested_keys: Sequence[str]) -> Tuple[Dict[str, ClassificationRecord], List[str]]:
        results: Dict[str, ClassificationRecord] = {}
        missing: List[str] = list(requested_keys)
        info = payload.get("Hierarchies") or payload.get("InformationList", {}).get("Information", [])
        if not info:
            # No data returned; leave everything marked missing.
            return results, missing

        def record_from_node(node: Dict) -> Optional[Tuple[str, ClassificationRecord]]:
            sources = node.get("Source") or []
            identifiers = node.get("InChIKey") or node.get("inchikey") or []
            if isinstance(identifiers, str):
                identifiers = [identifiers]
            class_name = node.get("Name") or node.get("NodeName")
            class_id = node.get("CID") or node.get("TaxonomyNodeID")
            url = None
            if node.get("URL"):
                url = node["URL"]
            if node.get("Reference"):
                url = node["Reference"]
            for key in identifiers:
                normalized = key.upper()
                if normalized in requested_keys and class_name:
                    return normalized, ClassificationRecord(
                        inchikey=normalized,
                        classification_name=class_name,
                        classification_id=str(class_id) if class_id is not None else None,
                        classification_url=url,
                        source="PubChem",
                        note=None,
                    )
            return None

        def walk_nodes(nodes: Iterable[Dict]) -> None:
            for node in nodes:
                if isinstance(node, dict):
                    candidate = record_from_node(node)
                    if candidate:
                        key, record = candidate
                        if key not in results:
                            results[key] = record
                    child_nodes = node.get("Node") or node.get("Children") or []
                    if isinstance(child_nodes, dict):
                        child_nodes = [child_nodes]
                    if child_nodes:
                        walk_nodes(child_nodes)

        if isinstance(info, dict):
            info = [info]
        walk_nodes(info)

        missing = [key for key in requested_keys if key not in results]
        return results, missing


class ChebiClient(BaseClient):
    SEARCH_URL = OLS_SEARCH_URL
    _ontology_cache: Optional[ChebiOntology] = None
    _ontology_path: Optional[Path] = None

    def __init__(
        self,
        session: requests.Session,
        rate_limiter: RateLimiter,
        *,
        obo_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(session, rate_limiter, **kwargs)
        self.obo_path = obo_path or DEFAULT_CHEBI_OBO
        if ChebiClient._ontology_cache is None or ChebiClient._ontology_path != self.obo_path:
            ChebiClient._ontology_cache = ChebiOntology(self.obo_path)
            ChebiClient._ontology_path = self.obo_path
        self._ontology = ChebiClient._ontology_cache

    def bulk_fetch(
        self, inchikeys: Sequence[str]
    ) -> Tuple[Dict[str, ClassificationRecord], List[str], Dict[str, str]]:
        records: Dict[str, ClassificationRecord] = {}
        missing: List[str] = []
        errors: Dict[str, str] = {}
        for key in inchikeys:
            record = self.fetch(key)
            if record.classification_name:
                records[key] = record
            else:
                missing.append(key)
                if record.note:
                    errors[key] = record.note
        return records, missing, errors

    def fetch(self, inchikey: str) -> ClassificationRecord:
        doc, search_error = self._search_by_inchikey(inchikey)
        if doc is None:
            return ClassificationRecord(
                inchikey=inchikey,
                classification_name=None,
                source="ChEBI",
                note=search_error or "No matching ChEBI term",
            )
        obo_id = doc.get("obo_id")
        if not obo_id:
            return ClassificationRecord(
                inchikey=inchikey,
                classification_name=None,
                source="ChEBI",
                note="Search result missing OBO identifier",
            )
        hierarchies = self._ontology.hierarchies(obo_id)
        if not hierarchies:
            return ClassificationRecord(
                inchikey=inchikey,
                classification_name=None,
                source="ChEBI",
                note="No parent classification available",
            )
        primary_path = hierarchies[0]
        primary = primary_path[0] if primary_path else None
        classification_name = primary.get("name") if primary else None
        classification_id = primary.get("id") if primary else None
        other_classes = [
            f"{path[0]['name']} ({path[0]['id']})"
            for path in hierarchies[1:]
            if path
        ]
        url = None
        if classification_id:
            url = f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={classification_id}"
        return ClassificationRecord(
            inchikey=inchikey,
            classification_name=classification_name,
            classification_id=classification_id,
            classification_url=url,
            source="ChEBI",
            note=None,
            other_classes="; ".join(other_classes) if other_classes else None,
            hierarchies=hierarchies,
        )

    def _search_by_inchikey(self, inchikey: str) -> Tuple[Optional[Dict], Optional[str]]:
        params = {
            "q": inchikey,
            "ontology": "chebi",
            "exact": "true",
            "rows": 1,
        }
        response = self._request("GET", self.SEARCH_URL, params=params)
        if response is None:
            return None, "Search request failed"
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return None, "Invalid JSON response"
        docs = payload.get("response", {}).get("docs") or []
        return (docs[0], None) if docs else (None, "No matching ChEBI term")



def read_compound_inputs(path: Path) -> List[NormalizedCompound]:
    compounds: List[NormalizedCompound] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            cleaned = raw.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            try:
                normalized = normalize_identifier(cleaned)
            except Exception as exc:
                sys.stderr.write(
                    f"[warn] Skipping line {line_number}: {cleaned} (reason: {exc})\n"
                )
                continue
            compounds.append(normalized)
    return compounds


def write_results(path: Optional[Path], records: Sequence[Dict[str, Any]], fmt: str = "json") -> None:
    fieldnames = ["inchikey", "input", "classifications", "notes"]
    fmt = (fmt or "json").lower()
    if fmt == "json":
        payload = list(records)
        if path is None:
            json.dump(payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
        return
    def row_from_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "inchikey": bundle.get("inchikey"),
            "input": json.dumps(bundle.get("input"), ensure_ascii=False),
            "classifications": json.dumps(bundle.get("classifications"), ensure_ascii=False),
            "notes": " | ".join(bundle.get("notes", [])),
        }

    if path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(row_from_bundle(record))
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(row_from_bundle(record))


def classify_inchikeys(
    compounds: Sequence[NormalizedCompound],
    classyfire_client: ClassyFireClient,
    fiehnlab_client: FiehnLabClient,
    chebi_client: Optional[ChebiClient] = None,
    pubchem_client: Optional[PubChemClient] = None,
    pubchem_batch_size: int = 10,
    enabled_services: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    inchikeys = [compound.inchikey for compound in compounds]
    notes: Dict[str, List[str]] = defaultdict(list)
    enabled = {svc.lower() for svc in (enabled_services or [])}
    service_results: Dict[str, Dict[str, ClassificationRecord]] = defaultdict(dict)

    def log_record(stage: str, record: ClassificationRecord) -> None:
        service_results[stage][record.inchikey] = record

    def log_event(stage: str, key: str, message: str) -> None:
        notes[key].append(f"{stage.title()}: {message}")

    pending_keys = [*inchikeys]
    if "classyfire" not in enabled and "fiehnlab" in enabled:
        fiehn_records, _, fiehn_errors = fiehnlab_client.bulk_fetch(inchikeys)
        for key, record in fiehn_records.items():
            log_record("fiehnlab", record)
        for key, message in fiehn_errors.items():
            log_event("fiehnlab", key, message)
        pending_keys = [key for key in inchikeys if key not in fiehn_records]

    if "classyfire" in enabled:
        classyfire_records, classyfire_missing, classyfire_errors = classyfire_client.bulk_fetch(inchikeys)
        for key, record in classyfire_records.items():
            log_record("classyfire", record)
        for key, message in classyfire_errors.items():
            log_event("classyfire", key, message)
        pending_keys = [key for key in inchikeys if key not in classyfire_records]
        missing_after_primary = [key for key in classyfire_missing if key in pending_keys]

        if missing_after_primary:
            fiehn_records, _, fiehn_errors = fiehnlab_client.bulk_fetch(missing_after_primary)
            for key, record in fiehn_records.items():
                log_record("fiehnlab", record)
            for key, message in fiehn_errors.items():
                log_event("fiehnlab", key, message)
            pending_keys = [
                key
                for key in inchikeys
                if key not in classyfire_records and key not in fiehn_records
            ]

    if pubchem_client and "pubchem" in enabled and pending_keys:
        pubchem_records: Dict[str, ClassificationRecord] = {}
        pubchem_missing: List[str] = []
        pubchem_errors: Dict[str, str] = {}
        for chunk in chunked(pending_keys, pubchem_batch_size):
            records, missing, errors = pubchem_client.bulk_fetch(chunk)
            pubchem_records.update(records)
            pubchem_missing.extend(missing)
            pubchem_errors.update(errors)
        for key, message in pubchem_errors.items():
            log_event("pubchem", key, message)
        for key, record in pubchem_records.items():
            log_record("pubchem", record)
        pending_keys = [key for key in inchikeys if key not in pubchem_records]

    if chebi_client and "chebi" in enabled:
        chebi_records, chebi_missing, chebi_errors = chebi_client.bulk_fetch(inchikeys)
        for key, message in chebi_errors.items():
            log_event("chebi", key, message)
        for key, record in chebi_records.items():
            log_record("chebi", record)

    bundles: List[Dict[str, Any]] = []
    for entry in compounds:
        key = entry.inchikey
        classifications: Dict[str, Dict[str, Any]] = {}
        for service_name, service_map in service_results.items():
            record = service_map.get(key)
            if record:
                record_dict = prune_nulls(asdict(record))
                if not record_dict:
                    continue
                if service_name in {"classyfire", "fiehnlab"}:
                    classifications.setdefault("chemont", []).append(record_dict)
                else:
                    classifications[service_name] = record_dict
        if not classifications:
            log_event("final", key, "No classification returned by enabled services")
        bundle = {
            "inchikey": key,
            "input": {
                "raw": entry.raw_input,
                "format": entry.input_format,
                "smiles": entry.smiles,
                "inchi": entry.inchi,
            },
            "classifications": classifications,
            "notes": notes.get(key) or [],
        }
        bundles.append(bundle)
    return bundles


def build_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify compounds by InChIKey, SMILES, or InChI. Select at least one "
            "service with --services."
        )
    )
    parser.add_argument("input", help="Path to a text file containing one compound identifier per line.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file. Defaults to stdout.",
    )
    parser.add_argument(
        "--enable-pubchem",
        action="store_true",
        default=False,
        help="Enable PubChem fallback in addition to the selected services.",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        required=True,
        choices=["classyfire", "fiehnlab", "chebi", "pubchem"],
        metavar="SERVICE",
        help=(
            "Classification service(s) to use. Choose from: classyfire, fiehnlab, "
            "chebi, pubchem. Selecting classyfire automatically tries FiehnLab for "
            "missing ChemOnt results."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json"],
        default=DEFAULT_OUTPUT_FORMAT,
        help="Output format (default: json).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    service_list = list(args.services)
    service_list = [str(s).strip().lower() for s in service_list if str(s).strip()]
    if args.enable_pubchem and "pubchem" not in service_list:
        service_list.append("pubchem")
    deduped_services: List[str] = []
    for svc in service_list:
        if svc not in deduped_services:
            deduped_services.append(svc)
    service_list = deduped_services

    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"[error] Input file not found: {input_path}\n")
        return 1
    compounds = read_compound_inputs(input_path)
    if not compounds:
        sys.stderr.write("[error] No valid compound identifiers found in input file.\n")
        return 1

    classyfire_rate = CLASSYFIRE_RATE_SECONDS
    classyfire_poll_interval = max(CLASSYFIRE_POLL_INTERVAL_SECONDS, 1.0)
    classyfire_max_wait_cfg = CLASSYFIRE_MAX_WAIT_SECONDS
    classyfire_max_wait = max(classyfire_poll_interval, classyfire_max_wait_cfg)
    classyfire_batch_size = int(max(1, min(CLASSYFIRE_BATCH_SIZE, 1000)))
    fiehnlab_rate = FIEHNLAB_RATE_SECONDS
    chebi_rate = CHEBI_RATE_SECONDS
    pubchem_rate = PUBCHEM_RATE_SECONDS
    pubchem_minute_cap = PUBCHEM_MINUTE_CAP
    pubchem_batch_size = PUBCHEM_BATCH_SIZE
    max_retries = MAX_RETRIES
    user_agent = DEFAULT_USER_AGENT

    session = build_session(user_agent)
    classyfire_post_interval = max(classyfire_rate, 0.1)

    classyfire_client = ClassyFireClient(
        session=session,
        rate_limiter=RateLimiter(max_calls=1, interval=classyfire_post_interval),
        batch_size=classyfire_batch_size,
        poll_interval=classyfire_poll_interval,
        max_wait=classyfire_max_wait,
        max_retries=max_retries,
    )
    fiehnlab_client = FiehnLabClient(
        session=session,
        rate_limiter=RateLimiter(max_calls=1, interval=max(fiehnlab_rate, 0.1)),
        max_retries=max_retries,
    )
    pubchem_client: Optional[PubChemClient] = None
    if "pubchem" in service_list:
        pubchem_short = RateLimiter(max_calls=1, interval=max(pubchem_rate, 0.05))
        pubchem_long = RateLimiter(max_calls=max(1, pubchem_minute_cap), interval=60.0)
        pubchem_client = PubChemClient(
            session=session,
            rate_limiter=CompositeRateLimiter([pubchem_short, pubchem_long]),
            max_retries=max_retries,
        )
    chebi_client: Optional[ChebiClient] = None
    if "chebi" in service_list:
        try:
            chebi_client = ChebiClient(
                session=session,
                rate_limiter=RateLimiter(max_calls=1, interval=max(chebi_rate, 0.1)),
                max_retries=max_retries,
            )
        except FileNotFoundError as exc:
            sys.stderr.write(f"[error] {exc}\n")
            sys.stderr.write(
                "[error] Download chebi.obo from https://www.ebi.ac.uk/chebi/downloads "
                f"and place it at {DEFAULT_CHEBI_OBO}, or run without the chebi service.\n"
            )
            return 1

    records = classify_inchikeys(
        compounds=compounds,
        classyfire_client=classyfire_client,
        fiehnlab_client=fiehnlab_client,
        chebi_client=chebi_client,
        pubchem_client=pubchem_client,
        pubchem_batch_size=max(1, pubchem_batch_size),
        enabled_services=service_list,
    )

    output_path = Path(args.output) if args.output else None
    write_results(output_path, records, fmt=args.output_format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
