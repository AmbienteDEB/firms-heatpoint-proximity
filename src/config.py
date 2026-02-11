import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

from dotenv import load_dotenv
from pathlib import Path

# ----------------------------
# Configuración
# ----------------------------

@dataclass(frozen=True)
class Config:
    results_path: str
    # FIRMS – DESCARGA DE PUNTOS DE CALOR
    map_key: str
    sources: List[str]
    days: int
    date: Optional[str]
    bbox: Tuple[float, float, float, float]  # west,south,east,north
    # POINTS VS POLYGONS (ANÁLISIS ESPACIAL)
    points_crs: str
    forced_polygons_crs: str
    polygons_path: Path
    polygons_name_field: str
    buffer_km: float
    metric_crs: str
    # GENERACIÓN DE IMÁGENES POR POLÍGONO
    tile_zoom: int
    dpi: int
    icon_path: Path
    icon_zoom: float
    buffer_icon_path: Optional[Path]
    buffer_icon_zoom: Optional[float]


    @staticmethod
    def from_env() -> "Config":
        load_dotenv()

        # CARPETA PARA GUARDAR LOS RESULTADOS
        results_path = (os.getenv("RESULTS_PATH") or "").strip()
        if not results_path:
            raise ValueError("Falta RESULTS_PATH en el .env")

        # FIRMS – DESCARGA DE PUNTOS DE CALOR
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

        # POINTS VS POLYGONS (ANÁLISIS ESPACIAL)

        points_crs = os.getenv("POINTS_CRS", "")
        if not points_crs:
            raise ValueError("POINTS_CRS debe establecerse en el .env (Generalmente el valor debe ser EPSG:4326)")

        forced_polygons_crs = os.getenv("FORCED_POLYGONS_CRS")
        if forced_polygons_crs:
            print(f"Se ha forzado el CRS a la capa de poligonos a {points_crs}")

        polygons_path = _validate_polygons_path(
            (os.getenv("POLYGONS_PATH") or "").strip(),
            forced_polygons_crs
        )

        polygons_name_field = os.getenv("POLYGONS_NAME_FIELD", "")
        if not polygons_name_field:
            raise ValueError("POLYGONS_NAME_FIELD debe establecerse en el .env")

        buffer_km = float(os.getenv("BUFFER_KM", "-1"))
        if not buffer_km > 0:
            raise ValueError("BUFFER_KM debe establecerce en el .env")

        metric_crs = os.getenv("METRIC_CRS", "")
        if not metric_crs:
            raise ValueError("METRIC_CRS debe establecerse en el .env (Generalmente el valor debe ser EPSG:32616)")

        # GENERACIÓN DE IMÁGENES POR POLÍGONO

        tile_zoom = int((os.getenv("TILE_ZOOM") or "12").strip())
        if not (0 <= tile_zoom <= 22):
            raise ValueError("TILE_ZOOM debe estar entre 0 y 22 (zoom típico de tiles web).")

        # Resolución de salida
        dpi_raw = (os.getenv("DPI") or "300").strip()

        try:
            dpi = int(dpi_raw)
        except ValueError:
            raise ValueError("DPI debe ser un número entero (ej. 300)")

        if not (72 <= dpi <= 1200):
            raise ValueError("DPI debe estar entre 72 y 1200")

        icon_path = _validate_icon_path((os.getenv("ICON_PATH") or "").strip())

        icon_zoom = float((os.getenv("ICON_ZOOM") or "1.0").strip())
        if not (icon_zoom > 0):
            raise ValueError("ICON_ZOOM debe ser > 0 (ej. 1.0).")

        # ICONO PARA BUFFER (OPCIONAL)
        buffer_icon_path_raw = (os.getenv("BUFFER_ICON_PATH") or "").strip()
        buffer_icon_path = _validate_optional_icon(buffer_icon_path_raw)

        buffer_icon_zoom_raw = (os.getenv("BUFFER_ICON_ZOOM") or "").strip()

        if buffer_icon_path and not buffer_icon_zoom_raw:
            raise ValueError("Si defines BUFFER_ICON_PATH, debes definir BUFFER_ICON_ZOOM")

        if buffer_icon_zoom_raw:
            buffer_icon_zoom = float(buffer_icon_zoom_raw)
            if not buffer_icon_zoom > 0:
                raise ValueError("BUFFER_ICON_ZOOM debe ser > 0")
        else:
            buffer_icon_zoom = None


        return Config(
            results_path=results_path,
            map_key=map_key,
            sources=parse_sources(sources_raw),
            days=days,
            date=date,
            bbox=parse_bbox(bbox_raw),
            points_crs=points_crs,
            forced_polygons_crs=forced_polygons_crs,
            polygons_path=polygons_path,
            polygons_name_field=polygons_name_field,
            buffer_km=buffer_km,
            metric_crs=metric_crs,
            tile_zoom=tile_zoom,
            dpi=dpi,
            icon_path=icon_path,
            icon_zoom=icon_zoom,
            buffer_icon_path=buffer_icon_path,
            buffer_icon_zoom=buffer_icon_zoom
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

# =========================
# Función privada de validación
# =========================

def _validate_polygons_path(path_raw: str, forced_crs: Optional[str]) -> Path:
    """
    Valida que la ruta de polígonos:
    - Exista
    - Sea archivo
    - Tenga extensión válida
    - Si es shapefile, tenga archivos sidecar mínimos
    """

    if not path_raw:
        raise ValueError("Falta POLYGONS_PATH en el .env")

    polygons_path = Path(path_raw)

    if not polygons_path.exists():
        raise ValueError(f"POLYGONS_PATH no existe: {polygons_path}")

    if not polygons_path.is_file():
        raise ValueError(f"POLYGONS_PATH no es un archivo válido: {polygons_path}")

    allowed_ext = {".gpkg", ".geojson", ".json", ".shp"}
    ext = polygons_path.suffix.lower()

    if ext not in allowed_ext:
        raise ValueError(
            f"POLYGONS_PATH debe ser uno de {sorted(allowed_ext)}. Recibido: '{ext}'"
        )

    # Validación adicional si es shapefile
    if ext == ".shp":
        required_sidecars = [".dbf", ".shx"]
        missing = [
            polygons_path.with_suffix(s)
            for s in required_sidecars
            if not polygons_path.with_suffix(s).exists()
        ]

        if missing:
            missing_str = ", ".join(str(p) for p in missing)
            raise ValueError(
                f"Shapefile incompleto. Faltan archivos requeridos: {missing_str}"
            )

        prj_file = polygons_path.with_suffix(".prj")
        if not prj_file.exists() and not forced_crs:
            raise ValueError(
                "El shapefile no tiene archivo .prj y no se definió FORCED_POLYGONS_CRS."
            )

    return polygons_path

def _validate_icon_path(path_raw: str) -> Path:
    """
    Valida que la ruta del icono:
    - Exista
    - Sea archivo
    - Tenga extensión de imagen soportada
    """
    if not path_raw:
        raise ValueError("Falta ICON_PATH en el .env")

    icon_path = Path(path_raw)

    if not icon_path.exists():
        raise ValueError(f"ICON_PATH no existe: {icon_path}")

    if not icon_path.is_file():
        raise ValueError(f"ICON_PATH no es un archivo válido: {icon_path}")

    allowed_ext = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
    ext = icon_path.suffix.lower()
    if ext not in allowed_ext:
        raise ValueError(f"ICON_PATH debe ser una imagen {sorted(allowed_ext)}. Recibido: '{ext}' ({icon_path})")

    return icon_path

def _validate_optional_icon(path_raw: str) -> Optional[Path]:
    """
    Valida icono opcional.
    Si está vacío, retorna None.
    Si existe, valida igual que icon normal.
    """
    if not path_raw:
        return None

    icon_path = Path(path_raw)

    if not icon_path.exists():
        raise ValueError(f"BUFFER_ICON_PATH no existe: {icon_path}")

    if not icon_path.is_file():
        raise ValueError(f"BUFFER_ICON_PATH no es archivo válido: {icon_path}")

    allowed_ext = {".png", ".jpg", ".jpeg", ".webp"}
    if icon_path.suffix.lower() not in allowed_ext:
        raise ValueError(
            f"BUFFER_ICON_PATH debe ser imagen {sorted(allowed_ext)}"
        )

    return icon_path
