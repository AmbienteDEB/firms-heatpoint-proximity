#!/usr/bin/env python3
"""
Lee el CSV de hotspots (ruta tomada desde .env) y lo convierte a Shapefile.
Guarda el .shp en una carpeta con el mismo nombre del archivo (sin extensión).

Requisitos (recomendado):
  pip install python-dotenv geopandas pyogrio

Notas:
- Asume que el CSV tiene columnas: latitude, longitude (formato FIRMS típico).
- Salida en EPSG:4326 (WGS84).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

def get_csv_path_from_env() -> Path:
    load_dotenv()
    csv_path = (os.getenv("FIRMS_OUT") or "").strip()
    if not csv_path:
        raise ValueError("Falta FIRMS_OUT en el .env (ruta/archivo del CSV de salida).")
    return Path(csv_path)

def ensure_columns(df, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"El CSV no contiene columnas requeridas: {missing}. "
            "Se esperan 'latitude' y 'longitude'."
        )

def main() -> int:
    try:
        import geopandas as gpd
        import pandas as pd
    except Exception as e:
        print(
            "✖ Faltan dependencias. Instala con:\n"
            "  pip install python-dotenv geopandas pyogrio\n",
            file=sys.stderr,
        )
        print(f"Detalle: {e}", file=sys.stderr)
        return 1

    try:
        csv_path = get_csv_path_from_env()

        # Permite que FIRMS_OUT sea relativo; se resuelve respecto al cwd
        csv_path = csv_path.expanduser().resolve()

        if not csv_path.exists():
            raise FileNotFoundError(f"No existe el CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        ensure_columns(df, ["latitude", "longitude"])

        # Carpeta con el mismo nombre del archivo (sin extensión)
        out_dir = csv_path.parent / csv_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        shp_path = out_dir / f"{csv_path.stem}.shp"

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )

        # Shapefile tiene limitaciones de nombres/longitud de campos; geopandas manejará truncamientos.
        gdf.to_file(shp_path, driver="ESRI Shapefile")

        print("✔ Conversión completada")
        print(f"CSV: {csv_path}")
        print(f"Salida (carpeta): {out_dir}")
        print(f"Shapefile: {shp_path}")
        return 0

    except Exception as e:
        print(f"✖ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
