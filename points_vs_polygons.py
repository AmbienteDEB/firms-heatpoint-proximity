#!/usr/bin/env python3
"""
Lista puntos (CSV FIRMS) que caen:
- dentro de un polígono, o
- dentro de un buffer (por defecto 1 km) alrededor del polígono.

Incluye en la salida:
- poly_fid  -> FID / índice del polígono
- area_nombre
- relation  -> inside | buffer
- distance_km -> distancia al polígono ORIGINAL

Configuración por .env
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv


# ----------------------------
# Configuración desde .env
# ----------------------------

def load_env():
    load_dotenv()

    cfg = {
        "csv_points": os.getenv("FIRMS_OUT"),
        "polygons_path": os.getenv("POLYGONS_PATH"),
        "name_field": os.getenv("POLYGONS_NAME_FIELD", "nombre"),
        "buffer_km": float(os.getenv("BUFFER_KM", "1")),
        "points_crs": os.getenv("POINTS_CRS", "EPSG:4326"),
        "polygons_crs": os.getenv("POLYGONS_CRS"),
        "metric_crs": os.getenv("METRIC_CRS", "EPSG:32616"),
        "output_csv": os.getenv("OUTPUT_CSV", "points_area_buffer.csv"),
    }

    for k in ("csv_points", "polygons_path"):
        if not cfg[k]:
            raise ValueError(f"Falta variable requerida en .env: {k}")

    return cfg


# ----------------------------
# Lectura de datos
# ----------------------------

def read_points(csv_path: str, crs: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)

    if not {"latitude", "longitude"}.issubset(df.columns):
        raise ValueError("El CSV debe contener columnas 'latitude' y 'longitude'")

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=crs,
    )

    gdf["_pt_id"] = range(len(gdf))
    return gdf


def read_polygons(path: str, name_field: str, force_crs: Optional[str]) -> gpd.GeoDataFrame:
    polys = gpd.read_file(path)

    if polys.empty:
        raise ValueError("El shapefile de polígonos está vacío")

    if force_crs:
        polys = polys.set_crs(force_crs, allow_override=True)

    if polys.crs is None:
        raise ValueError(
            "El shapefile de polígonos no tiene CRS definido. "
            "Define POLYGONS_CRS en el .env"
        )

    if name_field not in polys.columns:
        raise ValueError(
            f"No existe el campo '{name_field}' en el shapefile. "
            f"Campos disponibles: {list(polys.columns)}"
        )

    polys = polys[[name_field, "geometry"]].copy()
    polys = polys.rename(columns={name_field: "area_nombre"})

    # FID real del shapefile (índice)
    polys["poly_fid"] = polys.index

    return polys


# ----------------------------
# Análisis espacial
# ----------------------------

def compute_inside_or_buffer(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    metric_crs: str,
    buffer_km: float,
) -> pd.DataFrame:

    pts_m = points.to_crs(metric_crs)
    polys_m = polygons.to_crs(metric_crs)

    buffer_m = buffer_km * 1000.0

    # Distancia mínima al polígono original
    nearest = gpd.sjoin_nearest(
        pts_m[["_pt_id", "geometry"]],
        polys_m[["poly_fid", "area_nombre", "geometry"]],
        how="left",
        distance_col="distance_m",
    )

    # Puntos dentro del polígono
    inside = gpd.sjoin(
        pts_m[["_pt_id", "geometry"]],
        polys_m[["poly_fid", "area_nombre", "geometry"]],
        predicate="within",
        how="left",
    )

    inside_hits = inside.dropna(subset=["poly_fid"]).copy()
    inside_hits["relation"] = "inside"
    inside_hits["distance_m"] = 0.0
    inside_hits = inside_hits[["_pt_id", "poly_fid", "area_nombre", "relation", "distance_m"]]
    inside_hits = inside_hits.sort_values(["_pt_id"]).drop_duplicates("_pt_id")

    inside_ids = set(inside_hits["_pt_id"])

    # Puntos en buffer
    buffer_hits = nearest[~nearest["_pt_id"].isin(inside_ids)].copy()
    buffer_hits = buffer_hits[buffer_hits["distance_m"] <= buffer_m]
    buffer_hits["relation"] = "buffer"
    buffer_hits = buffer_hits[["_pt_id", "poly_fid", "area_nombre", "relation", "distance_m"]]

    # Unir resultados
    rel = pd.concat([inside_hits, buffer_hits], ignore_index=True)
    rel = rel.sort_values(["_pt_id", "distance_m"]).drop_duplicates("_pt_id")

    rel["distance_km"] = rel["distance_m"] / 1000.0
    return rel[["_pt_id", "poly_fid", "area_nombre", "relation", "distance_km"]]


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    try:
        cfg = load_env()

        points = read_points(cfg["csv_points"], cfg["points_crs"])
        polygons = read_polygons(cfg["polygons_path"], cfg["name_field"], cfg["polygons_crs"])

        rel = compute_inside_or_buffer(
            points,
            polygons,
            cfg["metric_crs"],
            cfg["buffer_km"],
        )

        # Solo puntos dentro del área o buffer
        output = (
            points
            .drop(columns="geometry")
            .merge(rel, on="_pt_id", how="inner")
            .drop(columns="_pt_id")
        )

        output.to_csv(cfg["output_csv"], index=False, encoding="utf-8")

        print("✔ Proceso completado")
        print(f"Puntos analizados: {len(points)}")
        print(f"Puntos dentro/buffer: {len(output)}")
        print(f"Salida: {cfg['output_csv']}")
        print("Campos añadidos: poly_fid, area_nombre, relation, distance_km")
        return 0

    except Exception as e:
        print(f"✖ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
