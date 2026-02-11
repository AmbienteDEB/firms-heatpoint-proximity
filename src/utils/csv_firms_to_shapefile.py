#!/usr/bin/env python3
"""
Lee el CSV de hotspots y lo convierte a Shapefile.
Guarda el .shp en una carpeta con el mismo nombre del archivo (sin extensión).

Requisitos (recomendado):
  pip install python-dotenv geopandas pyogrio

Notas:
- Asume que el CSV tiene columnas: latitude, longitude (formato FIRMS típico).
- Salida en EPSG:4326 (WGS84).
"""
import geopandas as gpd
import pandas as pd

from pathlib import Path


import sys

def ensure_columns(df, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"El CSV no contiene columnas requeridas: {missing}. "
            "Se esperan 'latitude' y 'longitude'."
        )

def convert_csv_firms_to_shapefile(csv_path:Path, output_dir:Path, points_crs) -> Path:
    # Permite que FIRMS_OUT sea relativo; se resuelve respecto al cwd
    csv_path = csv_path.expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    ensure_columns(df, ["latitude", "longitude"])

    shp_output_dir = output_dir / "hotspots"

    # Carpeta con el mismo nombre del archivo (sin extensión)
    if not shp_output_dir.exists():
        shp_output_dir.mkdir(parents=True, exist_ok=True)

    shp_path = shp_output_dir / f"hotspots.shp"

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=points_crs,
    )

    # Shapefile tiene limitaciones de nombres/longitud de campos; geopandas manejará truncamientos.
    gdf.to_file(shp_path, driver="ESRI Shapefile")

    return  shp_path
