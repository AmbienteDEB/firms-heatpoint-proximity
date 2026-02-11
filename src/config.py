import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

from dotenv import load_dotenv

# ----------------------------
# Configuración
# ----------------------------

@dataclass(frozen=True)
class Config:
    results_path: str
    map_key: str
    sources: List[str]
    days: int
    date: Optional[str]
    bbox: Tuple[float, float, float, float]  # west,south,east,north
    out_csv: str

    @staticmethod
    def from_env() -> "Config":
        load_dotenv()

        results_path = (os.getenv("RESULTS_PATH") or "").strip()
        if not results_path:
            raise ValueError("Falta RESULTS_PATH en el .env")

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
            results_path=results_path,
            map_key=map_key,
            sources=parse_sources(sources_raw),
            days=days,
            date=date,
            bbox=parse_bbox(bbox_raw),
            out_csv=out_csv,
        )

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