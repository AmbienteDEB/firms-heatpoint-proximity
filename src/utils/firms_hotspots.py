import csv
import requests
from typing import Optional, Tuple, List, Dict, Iterable
from pathlib import Path
from src.config import Config

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov"

# ----------------------------
# FIRMS HTTP
# ----------------------------

def http_get_text(url: str, timeout: int = 60) -> str:
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code}\nURL: {url}\nRespuesta: {resp.text[:400]}"
        )
    return resp.text


def build_firms_url(map_key: str, source: str, bbox: Tuple[float, float, float, float],
                    days: int, date: Optional[str]) -> str:
    west, south, east, north = bbox
    area = f"{west},{south},{east},{north}"
    url = f"{FIRMS_BASE}/api/area/csv/{map_key}/{source}/{area}/{days}"
    if date:
        url += f"/{date}"
    return url


# ----------------------------
# Parse / Merge / Dedup
# ----------------------------

def csv_text_to_rows(csv_text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Convierte el CSV FIRMS a: (header_fields, rows_dicts)
    """
    csv_text = (csv_text or "").strip()
    if not csv_text:
        return [], []

    reader = csv.DictReader(csv_text.splitlines())
    if not reader.fieldnames:
        return [], []

    rows = []
    for row in reader:
        # Normaliza keys y valores (sin romper columnas)
        clean = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        rows.append(clean)

    return list(reader.fieldnames), rows


def make_dedup_key(row: Dict[str, str], source: str) -> Tuple[str, ...]:
    """
    Clave de deduplicación robusta:
    - lat/lon (FIRMS usa 'latitude', 'longitude')
    - acq_date, acq_time (comunes en MODIS/VIIRS)
    - source (para no colapsar observaciones simultáneas de sensores distintos, si no quieres)
    """
    lat = (row.get("latitude") or "").strip()
    lon = (row.get("longitude") or "").strip()
    date = (row.get("acq_date") or "").strip()
    time = (row.get("acq_time") or "").strip()
    # Algunos productos pueden traer 'acq_datetime' u otros; si faltan, igual dedup por lat/lon/date/time
    return (lat, lon, date, time, source)


def collect_firms_records(cfg: Config) -> Tuple[List[str], List[Dict[str, str]]]:
    all_rows: List[Dict[str, str]] = []
    all_fields: List[str] = []

    # Para dedup y trazabilidad
    seen_keys = set()

    for source in cfg.sources:
        url = build_firms_url(cfg.map_key, source, cfg.bbox, cfg.days, cfg.date)
        text = http_get_text(url)

        fields, rows = csv_text_to_rows(text)

        # Si la respuesta trae solo header o viene vacía
        if not rows:
            print(f"ℹ Sin filas para source={source}")
            continue

        # Unifica campos: usa la unión de headers de todos los sources
        for f in fields:
            if f not in all_fields:
                all_fields.append(f)

        for r in rows:
            r["_source"] = source  # agrega columna para saber de qué fuente viene
            key = make_dedup_key(r, source)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_rows.append(r)

        print(f"✔ source={source}: {len(rows)} filas (acumulado dedup: {len(all_rows)})")

    # Asegura que _source quede en el header
    if "_source" not in all_fields:
        all_fields.append("_source")

    return all_fields, all_rows


# ----------------------------
# Exportación
# ----------------------------

def write_csv(path: Path, fields: List[str], rows: Iterable[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # Rellena faltantes con vacío para columnas nuevas
            out = {k: r.get(k, "") for k in fields}
            w.writerow(out)

