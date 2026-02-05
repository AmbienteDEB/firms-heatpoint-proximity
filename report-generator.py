#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generador-informe.py

Genera un informe PDF a partir de:
- CSV hotspots_vs_areas (OUTPUT_CSV)
- Imágenes por polígono (MAPS_OUT_DIR) creadas por el script de mapas

Incluye:
- Fecha de generación
- Resumen de puntos de calor
- Áreas con puntos de calor cercano (relation == 'buffer')
- Detalle por área:
    - Imagen
    - Cantidad de puntos
    - Información por punto (tabla)

.env esperado (mínimo):
  OUTPUT_CSV=hotspots_vs_areas.csv
  MAPS_OUT_DIR=maps_by_polygon
  BUFFER_KM=2

Opcional:
  PDF_OUT=reports/informe_hotspots_vs_areas.pdf
  # Si no se define, se genera: reports/informe_hotspots_vs_areas_YYYYMMDD_HHMMSS.pdf

Requisitos:
  pip install reportlab python-dotenv pandas
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader


# ----------------------------
# Configuración desde .env
# ----------------------------

def load_env() -> dict:
    load_dotenv()
    cfg = {
        "csv_path": (os.getenv("OUTPUT_CSV") or "").strip(),
        "maps_out_dir": (os.getenv("MAPS_OUT_DIR") or "maps_by_polygon").strip(),
        "pdf_out": (os.getenv("PDF_OUT") or "").strip(),
        "buffer_km": os.getenv("BUFFER_KM")
    }
    if not cfg["csv_path"]:
        raise ValueError("Falta OUTPUT_CSV en el .env")
    if not cfg["buffer_km"]:
        raise ValueError("Falta BUFFER_KM en el .env")
    return cfg


def safe_filename(s: str, max_len: int = 120) -> str:
    s = (s or "").strip() or "sin_nombre"
    s = re.sub(r"[^\w\-\. ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len]


# ----------------------------
# Utilidades de imágenes (FIX: too large on page)
# ----------------------------

def make_scaled_rl_image(image_path: Path, max_w_pts: float, max_h_pts: float) -> Image:
    """
    Retorna un reportlab.platypus.Image escalado para caber en (max_w_pts, max_h_pts)
    manteniendo aspecto. Unidades en puntos (pt).
    """
    ir = ImageReader(str(image_path))
    iw, ih = ir.getSize()  # pixeles (sirve para proporción)

    if not iw or not ih:
        return Image(str(image_path))

    scale = min(max_w_pts / float(iw), max_h_pts / float(ih))
    scale = min(scale, 1.0)  # no agrandar, solo reducir

    img = Image(str(image_path))
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    return img


def find_area_image(maps_dir: Path, poly_fid: int) -> Path | None:
    """
    Busca la imagen generada por tu script:
      fid_{fid}__{area}.png
    pero tolerante: fid_{fid}__*.png
    """
    if not maps_dir.exists():
        return None
    hits = sorted(maps_dir.glob(f"fid_{poly_fid}__*.png"))
    return hits[0] if hits else None


# ----------------------------
# Formateos comunes
# ----------------------------

def fmt_time_hhmm(acq_time) -> str:
    """Convierte HHMM (707, 1545, 30, 0, 707.0) a '07:07 AM'."""
    if pd.isna(acq_time):
        return ""
    try:
        t = int(float(acq_time))          # soporta 707.0
        hhmm = f"{t:04d}"                 # 707 -> "0707"
        return datetime.strptime(hhmm, "%H%M").strftime("%I:%M %p")
    except Exception:
        return str(acq_time)


def build_point_columns(df: pd.DataFrame) -> list[str]:
    """
    Selecciona columnas útiles sin romper si no existen.

    ✅ Cambiado: latitude + longitude se reemplazan por columna virtual 'coords'
       que renderiza "Lat,Lon" y link a Google Maps.
    """
    preferred_cols = [
        # "latitude",   Ya no se utiliza
        # "longitude",  Ya no se utiliza
        "coords",       # <-- virtual
        "acq_date",
        "acq_time",
        # "confidence",
        "frp",
        # "bright_ti4",
        # "bright_ti5",
        # "daynight",
        # "satellite",
        # "instrument",
        # "relation",
        "distance_km",
        # "_source",
    ]

    cols = []
    for c in preferred_cols:
        if c == "coords":
            # Solo mostrar si existen lat/lon en el CSV
            if "latitude" in df.columns and "longitude" in df.columns:
                cols.append("coords")
        else:
            if c in df.columns:
                cols.append(c)
    return cols


def humanize_col(c: str) -> str:
    mapping = {
        "coords": "Coordenadas",
        "latitude": "Lat",
        "longitude": "Lon",
        "acq_date": "Fecha",
        "acq_time": "Hora",
        "confidence": "Conf",
        "frp": "FRP",
        "bright_ti4": "T4",
        "bright_ti5": "T5",
        "daynight": "D/N",
        "satellite": "Sat",
        "instrument": "Instr",
        "relation": "Relación",
        "distance_km": "Dist(km)",
        "_source": "Fuente",
    }
    return mapping.get(c, c)


def format_point_value(col: str, v, row: dict, styles) -> object:
    """
    Devuelve el valor formateado. Puede retornar str o Paragraph.

    ✅ coords: "Lat,Lon" clickeable a Google Maps.
    """
    if col == "coords":
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            return ""
        try:
            lat_f = float(lat)
            lon_f = float(lon)
            text = f"{lat_f:.6f},{lon_f:.6f}"
            url = f"https://www.google.com/maps?q={lat_f:.6f},{lon_f:.6f}"
            # Paragraph con link clickeable (azul y subrayado)
            return Paragraph(
                f'<a href="{url}"><u><font color="blue">{text}</font></u></a>',
                styles["Small"]
            )
        except Exception:
            return ""

    if pd.isna(v):
        return ""
    try:
        if col in ("latitude", "longitude"):
            return f"{float(v):.6f}"
        if col in ("frp", "bright_ti4", "bright_ti5"):
            return f"{float(v):.2f}"
        if col == "distance_km":
            return f"{float(v):.3f}"
        if col == "acq_time":
            return fmt_time_hhmm(v)
    except Exception:
        pass
    return str(v)


def compute_col_widths(header: list[str], total_width_pts: float) -> list[float]:
    """
    Asigna anchos relativos por columna para que la tabla sea legible.
    """
    weights = []
    for c in header:
        if c in ("area_nombre",):
            weights.append(3.0)
        elif c in ("coords",):
            weights.append(2.4)
        elif c in ("acq_date",):
            weights.append(1.7)
        elif c in ("acq_time", "confidence", "daynight", "relation"):
            weights.append(1.1)
        elif c in ("distance_km", "frp", "bright_ti4", "bright_ti5"):
            weights.append(1.2)
        else:
            weights.append(1.3)

    s = sum(weights) if weights else 1.0
    widths = [max(55.0, total_width_pts * (w / s)) for w in weights]  # mínimo 45pt
    # normaliza de nuevo para no pasarse del total
    factor = total_width_pts / sum(widths)
    widths = [w * factor for w in widths]
    return widths


# ----------------------------
# PDF Builder
# ----------------------------

def build_pdf(csv_path: Path, maps_dir: Path, pdf_out: Path, buffer_km: float) -> Path:
    df = pd.read_csv(csv_path)

    required = {"latitude", "longitude", "poly_fid", "area_nombre", "relation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El CSV no tiene columnas requeridas: {sorted(missing)}")

    df = df.copy()
    df["area_nombre"] = df["area_nombre"].fillna("Sin nombre")
    df["relation"] = df["relation"].fillna("")
    df["poly_fid"] = pd.to_numeric(df["poly_fid"], errors="coerce").astype("Int64")

    # Métricas globales
    total_points = len(df)
    total_areas = int(df["poly_fid"].dropna().nunique())
    inside_points = int((df["relation"].str.lower() == "inside").sum())
    buffer_points = int((df["relation"].str.lower() == "buffer").sum())

    # Áreas con puntos cercanos (= buffer)
    areas_with_buffer = (
        df[df["relation"].str.lower() == "buffer"]
        .dropna(subset=["poly_fid"])
        .groupby(["poly_fid", "area_nombre"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )

    pdf_out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_out),
        pagesize=landscape(A4),
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
        title="Informe de Hotspots vs Áreas",
        author="Script automático",
        compression=1
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=18, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=11))

    story = []

    gen_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph("Informe Puntos de Calor en ANP", styles["H1x"]))
    story.append(Paragraph(f"<b>Fecha de generación:</b> {gen_dt}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Distancia buffer:</b> {buffer_km} km", styles["Small"]))
    story.append(Spacer(1, 10))

    # Resumen
    story.append(Paragraph("Resumen", styles["H2x"]))
    resumen_data = [
        ["Métrica", "Valor"],
        ["Puntos de calor", f"{total_points}"],
        ["Áreas con puntos de calor cercanos", f"{total_areas}"],
        ["Puntos dentro del área", f"{inside_points}"],
        ["Puntos cercanos", f"{buffer_points}"],
    ]
    resumen_tbl = Table(resumen_data, colWidths=[7 * cm, 6 * cm])
    resumen_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(resumen_tbl)
    story.append(Spacer(1, 12))

    # Áreas con puntos cercanos
    story.append(Paragraph("Áreas con puntos de calor cercano", styles["H2x"]))
    if areas_with_buffer.empty:
        story.append(Paragraph("No se encontraron puntos con relation=buffer.", styles["BodyText"]))
    else:
        data = [["Área", "Puntos cercanos"]]
        for _, r in areas_with_buffer.iterrows():
            data.append([str(r["area_nombre"]), str(int(r["size"]))])

        tbl = Table(data, colWidths=[18 * cm, 5 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(tbl)

    story.append(PageBreak())

    # Detalle por área
    story.append(Paragraph("Detalle por área", styles["H1x"]))

    # Ordenar por # puntos desc
    areas = (
        df.dropna(subset=["poly_fid"])
        .groupby(["poly_fid", "area_nombre"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )

    cols = build_point_columns(df)
    header_labels = [humanize_col(c) for c in cols]

    for _, a in areas.iterrows():
        fid = int(a["poly_fid"])
        area_name = str(a["area_nombre"])
        npts = int(a["size"])

        story.append(Paragraph(f"{area_name}", styles["H2x"]))
        story.append(Paragraph(f"<b>Cantidad de puntos:</b> {npts}", styles["BodyText"]))
        story.append(Spacer(1, 4))

        # Imagen del área (si existe) - escalada para caber en página
        img_path = find_area_image(maps_dir, fid)
        if img_path and img_path.exists():
            # doc.width / doc.height están en puntos
            max_w = doc.width
            # Reserva espacio vertical para títulos + espaciadores en esta página
            reserved_h = 2.5 * cm
            max_h = max(5 * cm, doc.height - reserved_h)

            img = make_scaled_rl_image(img_path, max_w_pts=max_w, max_h_pts=max_h)
            story.append(img)
            story.append(Spacer(1, 8))
        else:
            story.append(Paragraph(
                f"<i>Imagen no encontrada para FID {fid} en {maps_dir}.</i>",
                styles["Small"]
            ))
            story.append(Spacer(1, 8))

        # Para estabilidad: la tabla SIEMPRE inicia en página nueva
        story.append(PageBreak())

        # Tabla de puntos del área
        pts = df[df["poly_fid"] == fid].copy()

        table_data = [header_labels]
        for _, r in pts.iterrows():
            row_dict = r.to_dict()
            row = []
            for c in cols:
                if c == "coords":
                    row.append(format_point_value("coords", None, row_dict, styles))
                else:
                    row.append(format_point_value(c, row_dict.get(c, ""), row_dict, styles))
            table_data.append(row)

        total_table_width = doc.width  # puntos
        col_widths = compute_col_widths(cols, total_width_pts=total_table_width)

        pts_tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
        pts_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 2),
        ]))

        story.append(Paragraph(f"Puntos asociados a: {area_name}", styles["H2x"]))
        story.append(pts_tbl)
        story.append(PageBreak())

    doc.build(story)
    return pdf_out


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    try:
        cfg = load_env()

        csv_path = Path(cfg["csv_path"]).expanduser().resolve()
        maps_dir = Path(cfg["maps_out_dir"]).expanduser().resolve()
        buffer_km = float(cfg["buffer_km"])

        if not csv_path.exists():
            raise FileNotFoundError(f"No existe el CSV: {csv_path}")

        if cfg["pdf_out"]:
            pdf_out = Path(cfg["pdf_out"]).expanduser().resolve()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_out = (Path("reports") / f"informe_hotspots_vs_areas_{ts}.pdf").resolve()

        out = build_pdf(csv_path=csv_path, maps_dir=maps_dir, pdf_out=pdf_out, buffer_km=buffer_km)

        print("✔ Informe PDF generado")
        print(f"CSV: {csv_path}")
        print(f"Imágenes: {maps_dir}")
        print(f"Salida PDF: {out}")
        return 0

    except Exception as e:
        print(f"✖ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
