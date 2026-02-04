#!/usr/bin/env python3
"""
Filtra puntos (desde CSV FIRMS) que caen:
- dentro del polígono, O
- dentro de un buffer (por defecto 1 km) alrededor del polígono.

Para cada punto incluido:
- indica el 'area_nombre' (campo del shapefile, por defecto 'nombre')
- indica la 'relation' = inside | buffer
- indica 'distance_km' = distancia al polígono ORIGINAL (0 si está inside)

Exporta a CSV conservando TODAS las columnas originales del CSV FIRMS + campos añadidos.

Configuración por .env:
  FIRMS_OUT=hotspots_multi.csv
  POLYGONS_PATH=areas.shp
  POLYGONS_NAME_FIELD=nombre
  BUFFER_KM=1
  POINTS_CRS=EPSG:4326
  # POLYGONS_CRS=EPSG:4326   (solo si el shapefile no trae CRS)
  METRIC_CRS=EPSG:32616
  OUTPUT_CSV=hotspots_in_area_or_buffer.csv

Requisitos:
  pip install pandas geopandas shapely pyproj python-dotenv pyogrio
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
        "csv_points": (os.getenv("FIRMS_OUT") or "").strip(),
        "polygons_path": (os.getenv("POLYGONS_PATH") or "").strip(),
        "name_field": (os.getenv("POLYGONS_NAME_FIELD") or "nombre").strip(),
        "buffer_km": float((os.getenv("BUFFER_KM") or "1").strip()),
        "points_crs": (os.getenv("POINTS_CRS") or "EPSG:4326").strip(),
        "polygons_crs": (os.getenv("POLYGONS_CRS") or "").strip() or None,
        "metric_crs": (os.getenv("METRIC_CRS") or "EPSG:32616").strip(),
        "output_csv": (os.getenv("OUTPUT_CSV") or "points_in_area_or_buffer.csv").strip(),
    }

    for k in ("csv_points", "polygons_path"):
        if not cfg[k]:
            raise ValueError(f"Falta variable requerida en .env: {k}")

    if cfg["buffer_km"] <= 0:
        raise ValueError("BUFFER_KM debe ser > 0")

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
        raise ValueError("El archivo de polígonos está vacío")

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
    polys["_poly_id"] = range(len(polys))
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
    """
    Retorna tabla por punto seleccionado:
      _pt_id, area_nombre, relation(inside|buffer), distance_km (al polígono original)
    Si un punto cae dentro de múltiples polígonos/buffers, conserva la asignación
    con MENOR distancia al polígono original.
    """
    pts_m = points.to_crs(metric_crs)
    polys_m = polygons.to_crs(metric_crs)

    buffer_m = buffer_km * 1000.0

    # Distancia al polígono original (0 si está dentro)
    # sjoin_nearest nos da polígono más cercano y distancia mínima a su borde
    nearest = gpd.sjoin_nearest(
        pts_m[["_pt_id", "geometry"]],
        polys_m[["_poly_id", "area_nombre", "geometry"]],
        how="left",
        distance_col="distance_m",
    )

    # Clasificación inside/buffer/outside:
    # - inside: punto within(polígono)
    # - buffer: no está inside pero distance_m <= buffer_m
    # - outside: distance_m > buffer_m -> se excluye
    inside_flag = gpd.sjoin(
        pts_m[["_pt_id", "geometry"]],
        polys_m[["_poly_id", "area_nombre", "geometry"]],
        predicate="within",
        how="left",
    )

    # Map de puntos inside (puede haber solapes; tomamos el más "cercano" al polígono original: distancia 0)
    inside_map = inside_flag.dropna(subset=["_poly_id"]).copy()
    if not inside_map.empty:
        inside_map["distance_m"] = 0.0
        inside_map["relation"] = "inside"
        inside_map = inside_map[["_pt_id", "area_nombre", "relation", "distance_m"]]
        # si un punto cae en varios polígonos (solapados), quedarse con uno (primero)
        inside_map = inside_map.sort_values(["_pt_id", "area_nombre"]).drop_duplicates("_pt_id", keep="first")
    else:
        inside_map = pd.DataFrame(columns=["_pt_id", "area_nombre", "relation", "distance_m"])

    # Para los no-inside: usar nearest y filtrar por buffer
    inside_ids = set(inside_map["_pt_id"])
    nearest_not_inside = nearest[~nearest["_pt_id"].isin(inside_ids)].copy()

    # Filtrar a los que caen dentro del buffer (<= 1km)
    within_buffer = nearest_not_inside[nearest_not_inside["distance_m"] <= buffer_m].copy()
    within_buffer["relation"] = "buffer"
    within_buffer = within_buffer[["_pt_id", "area_nombre", "relation", "distance_m"]]

    # Unir inside + buffer
    rel = pd.concat([inside_map, within_buffer], ignore_index=True)

    # Si por alguna razón un punto aparece duplicado (p.ej. nearest ambiguo), quedarnos con menor distancia
    rel = rel.sort_values(["_pt_id", "distance_m"]).drop_duplicates("_pt_id", keep="first")

    rel["distance_km"] = rel["distance_m"].astype(float) / 1000.0
    return rel[["_pt_id", "area_nombre", "relation", "distance_km"]]


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    try:
        cfg = load_env()

        points = read_points(cfg["csv_points"], cfg["points_crs"])
        polygons = read_polygons(cfg["polygons_path"], cfg["name_field"], cfg["polygons_crs"])

        rel = compute_inside_or_buffer(
            points=points,
            polygons=polygons,
            metric_crs=cfg["metric_crs"],
            buffer_km=cfg["buffer_km"],
        )

        # Mantener solo puntos que están inside o dentro del buffer
        # (si un punto no aparece en rel, está fuera del área+buffer)
        selected = points.drop(columns="geometry").merge(rel, on="_pt_id", how="inner").drop(columns="_pt_id")

        selected.to_csv(cfg["output_csv"], index=False, encoding="utf-8")

        print("✔ Listado generado (solo puntos dentro del área o dentro del buffer)")
        print(f"CSV puntos: {cfg['csv_points']}")
        print(f"Polígonos: {cfg['polygons_path']}")
        print(f"Buffer: {cfg['buffer_km']} km")
        print(f"Salida: {cfg['output_csv']}")
        print(f"Puntos incluidos: {len(selected)}")
        print("Campos añadidos: area_nombre, relation (inside|buffer), distance_km (al polígono original)")
        return 0

    except Exception as e:
        print(f"✖ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
