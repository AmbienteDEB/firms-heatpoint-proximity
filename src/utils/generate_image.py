#!/usr/bin/env python3
"""
Genera UNA IMAGEN por polígono (poly_fid) con sus puntos asociados, sobre
mapa SATELITAL HÍBRIDO (imagen + etiquetas), usando ICONO PNG como marcador.

✅ Fix: "Can not put single artist in more than one figure"
No se reutiliza OffsetImage entre figuras; se carga el PNG como array y se crea
OffsetImage NUEVO dentro de cada figura/punto.

.env:
  POLYGONS_PATH=ruta/a/tu_multipoligonos.shp
  POLYGONS_NAME_FIELD=nombre
  # POLYGONS_CRS=EPSG:4326     (solo si el shape no trae CRS)

  BUFFER_KM=1
  TILE_ZOOM=13
  DPI=300

  FIRE_ICON_PATH=fire.png
  FIRE_ICON_ZOOM=0.06
  # Opcional para puntos en buffer:
  BUFFER_ICON_PATH=fire_buffer.png
  BUFFER_ICON_ZOOM=0.06

Requisitos:
  pip install pandas geopandas matplotlib contextily python-dotenv pyogrio shapely pyproj
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

from src.config import Config

def safe_filename(s: str, max_len: int = 120) -> str:
    s = (s or "").strip() or "sin_nombre"
    s = re.sub(r"[^\w\-\. ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len]


# ----------------------------
# Lectura insumos
# ----------------------------

def read_polygons(path: Path, name_field: str, force_crs: Optional[str]) -> gpd.GeoDataFrame:
    polys = gpd.read_file(path)
    if polys.empty:
        raise ValueError("El archivo de polígonos está vacío.")

    if force_crs:
        polys = polys.set_crs(force_crs, allow_override=True)

    if polys.crs is None:
        raise ValueError("El archivo de polígonos no tiene CRS. Define POLYGONS_CRS en el .env.")

    if name_field not in polys.columns:
        raise ValueError(f"No existe el campo '{name_field}' en el shapefile. Campos: {list(polys.columns)}")

    polys = polys[[name_field, "geometry"]].copy()
    polys = polys.rename(columns={name_field: "area_nombre"})
    polys["poly_fid"] = polys.index.astype(int)  # consistente con scripts previos
    return polys


def read_processed_points(csv_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)

    required = {"latitude", "longitude", "poly_fid", "area_nombre", "relation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El CSV procesado no tiene columnas requeridas: {sorted(missing)}")

    df["poly_fid"] = df["poly_fid"].astype(int)

    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )


# ----------------------------
# Basemap híbrido
# ----------------------------

def add_hybrid_basemap(ax, zoom: int) -> None:
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs="EPSG:3857",
                    attribution=False, zoom=zoom)
    # Labels encima
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldBoundariesAndPlaces, crs="EPSG:3857",
                        attribution=False, zoom=zoom)
    except Exception:
        pass


# ----------------------------
# Iconos PNG
# ----------------------------

def load_icon_array(icon_path: Path) -> "object":
    p = Path(icon_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"No existe el icono PNG: {p}")
    return mpimg.imread(str(p))


def plot_points_as_icons(
    ax,
    pts_3857: gpd.GeoDataFrame,
    inside_img_arr,
    inside_zoom: float,
    buffer_img_arr=None,
    buffer_zoom: Optional[float] = None,
) -> None:
    """
    Crea OffsetImage NUEVO por punto (evita reusar Artists entre figuras).
    """
    if buffer_zoom is None:
        buffer_zoom = inside_zoom

    for _, r in pts_3857.iterrows():
        rel = str(r.get("relation", "")).lower()
        if buffer_img_arr is not None and rel == "buffer":
            img_arr = buffer_img_arr
            zoom = buffer_zoom
        else:
            img_arr = inside_img_arr
            zoom = inside_zoom

        oi = OffsetImage(img_arr, zoom=zoom)  # NUEVO objeto por punto
        ab = AnnotationBbox(
            oi,
            (r.geometry.x, r.geometry.y),
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0,
        )
        ax.add_artist(ab)


# ----------------------------
# Plot por polígono
# ----------------------------

def plot_one_polygon_hybrid_png(
    poly_gdf_1row: gpd.GeoDataFrame,
    pts_gdf: gpd.GeoDataFrame,
    out_path: Path,
    pad_m: float,
    zoom: int,
    dpi: int,
    inside_img_arr,
    inside_icon_zoom: float,
    buffer_img_arr,
    buffer_icon_zoom: float,
) -> None:
    poly_3857 = poly_gdf_1row.to_crs(epsg=3857)
    pts_3857 = pts_gdf.to_crs(epsg=3857)

    # Extent con padding + bleed
    minx, miny, maxx, maxy = poly_3857.total_bounds
    minx -= pad_m
    miny -= pad_m
    maxx += pad_m
    maxy += pad_m

    epsx = (maxx - minx) * 0.003
    epsy = (maxy - miny) * 0.003

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor("none")
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    ax.set_xlim(minx - epsx, maxx + epsx)
    ax.set_ylim(miny - epsy, maxy + epsy)

    try:
        add_hybrid_basemap(ax, zoom=zoom)
    except Exception:
        pass

    # Borde polígono
    poly_3857.boundary.plot(ax=ax, linewidth=2)

    # Puntos con icono PNG
    plot_points_as_icons(
        ax=ax,
        pts_3857=pts_3857,
        inside_img_arr=inside_img_arr,
        inside_zoom=inside_icon_zoom,
        buffer_img_arr=buffer_img_arr,
        buffer_zoom=buffer_icon_zoom,
    )

    # Título overlay
    area = str(poly_3857["area_nombre"].iloc[0])
    fid = int(poly_3857["poly_fid"].iloc[0])
    ax.text(
        0.5, 0.98,
        f"{area}  |  Puntos={len(pts_3857)}",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=22, weight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.45, edgecolor="none")
    )

    ax.axis("off")
    ax.set_aspect("auto")
    ax.set_anchor("C")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def generate_images(cfg: Config, ouput_dir: Path, csv_processed_path: Path) -> Path:
    image_out_dir = ouput_dir / 'images'
    if not image_out_dir.exists():
        image_out_dir.mkdir(parents=True, exist_ok=True)

    polys = read_polygons(cfg.polygons_path, cfg.polygons_name_field, cfg.forced_polygons_crs)
    points = read_processed_points(csv_processed_path)

    poly_fids = sorted(points["poly_fid"].dropna().unique().tolist())
    if not poly_fids:
        raise ValueError("No hay poly_fid en el CSV procesado (¿está vacío o no coincide?).")

    # Cargar imágenes PNG como arrays (NO OffsetImage)
    inside_img_arr = load_icon_array(cfg.icon_path)
    buffer_img_arr = load_icon_array(cfg.buffer_icon_path) if cfg.buffer_icon_path else None

    pad_m = cfg.buffer_km * 1000.0
    polys_idx = polys.set_index("poly_fid", drop=False)

    total_imgs = 0
    for fid in poly_fids:
        if fid not in polys_idx.index:
            continue

        poly_1 = polys_idx.loc[[fid]]
        pts = points[points["poly_fid"] == fid].copy()

        area_name = safe_filename(str(poly_1["area_nombre"].iloc[0]))
        out_path = image_out_dir / f"fid_{fid}__{area_name}.png"

        plot_one_polygon_hybrid_png(
            poly_gdf_1row=poly_1,
            pts_gdf=pts,
            out_path=out_path,
            pad_m=pad_m,
            zoom=cfg.tile_zoom,
            dpi=cfg.dpi,
            inside_img_arr=inside_img_arr,
            inside_icon_zoom=cfg.icon_zoom,
            buffer_img_arr=buffer_img_arr,
            buffer_icon_zoom=cfg.buffer_icon_zoom,
        )
        total_imgs += 1

    print("✔ Mapas híbridos generados con iconos PNG")
    print(f"Salida imágenes: {image_out_dir}")
    print(f"Imágenes creadas: {total_imgs}")

    return image_out_dir
