#!/usr/bin/env python3
"""
Descarga puntos de calor de NASA FIRMS para un BBOX definido por el usuario,
consultando múltiples fuentes (sources) y exportando un solo CSV deduplicado.

Requisitos:
  pip install requests python-dotenv

.env requerido:
  FIRMS_MAP_KEY=...
  FIRMS_SOURCES=VIIRS_SNPP_NRT,MODIS_NRT,...
  FIRMS_BBOX=west,south,east,north
Opcionales:
  FIRMS_DAYS=1   (1..5)
  FIRMS_DATE=YYYY-MM-DD
  FIRMS_OUT=hotspots_multi.csv
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Iterable

import requests
from dotenv import load_dotenv

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov"


# ----------------------------
# Configuración
# ----------------------------

@dataclass(frozen=True)
class Config:
    map_key: str
    sources: List[str]
    days: int
    date: Optional[str]
    bbox: Tuple[float, float, float, float]  # west,south,east,north
    out_csv: str


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("FIRMS_BBOX debe tener 4 valores: west,south,east,north")

    west, south, east, north = map(float, parts)

    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("Longitudes fuera de rango (-180..180).")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError("Latitudes fuera de rango (-90..90).")
    if west >= east:
        raise ValueError("BBOX inválido: west debe ser < east.")
    if south >= north:
        raise ValueError("BBOX inválido: south debe ser < north.")

    return west, south, east, north


def parse_sources(raw: str) -> List[str]:
    sources = [s.strip() for s in raw.split(",") if s.strip()]
    if not sources:
        raise ValueError("FIRMS_SOURCES está vacío. Ej: VIIRS_SNPP_NRT,MODIS_NRT")
    # Dedup manteniendo orden
    seen = set()
    out = []
    for s in sources:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def load_config() -> Config:
    load_dotenv()

    map_key = (os.getenv("FIRMS_MAP_KEY") or "").strip()
    if not map_key:
        raise ValueError("Falta FIRMS_MAP_KEY en el .env")

    sources_raw = (os.getenv("FIRMS_SOURCES") or "").strip()
    if not sources_raw:
        raise ValueError("Falta FIRMS_SOURCES en el .env (lista separada por comas)")

    bbox_raw = (os.getenv("FIRMS_BBOX") or "").strip()
    if not bbox_raw:
        raise ValueError("Falta FIRMS_BBOX en el .env (west,south,east,north)")

    days = int((os.getenv("FIRMS_DAYS") or "1").strip())
    if not (1 <= days <= 5):
        raise ValueError("FIRMS_DAYS debe estar entre 1 y 5")

    date = (os.getenv("FIRMS_DATE") or "").strip() or None
    out_csv = (os.getenv("FIRMS_OUT") or "hotspots_multi.csv").strip()

    return Config(
        map_key=map_key,
        sources=parse_sources(sources_raw),
        days=days,
        date=date,
        bbox=parse_bbox(bbox_raw),
        out_csv=out_csv,
    )


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


def merge_multisource(cfg: Config) -> Tuple[List[str], List[Dict[str, str]]]:
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

def write_csv(path: str, fields: List[str], rows: Iterable[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # Rellena faltantes con vacío para columnas nuevas
            out = {k: r.get(k, "") for k in fields}
            w.writerow(out)


def main() -> int:
    try:
        cfg = load_config()
        fields, rows = merge_multisource(cfg)

        if not rows:
            raise RuntimeError("No se obtuvieron filas de FIRMS (revisa MAP_KEY, BBOX, DAYS/SOURCES).")

        write_csv(cfg.out_csv, fields, rows)

        print("\n=== RESUMEN ===")
        print(f"Archivo: {cfg.out_csv}")
        print(f"Sources: {', '.join(cfg.sources)}")
        print(f"Días: {cfg.days}" + (f" | Fecha: {cfg.date}" if cfg.date else " | Fecha: (más reciente)"))
        print(f"BBOX: {cfg.bbox} (west,south,east,north)")
        print(f"Filas exportadas (dedup): {len(rows)}")
        return 0

    except Exception as e:
        print(f"✖ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
