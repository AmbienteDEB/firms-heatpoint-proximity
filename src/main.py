import sys
from datetime import datetime
from pathlib import Path

from src.config import Config
from src.utils.firms_hotspots import collect_firms_records, write_csv
from src.utils.csv_firms_to_shapefile import convert_csv_firms_to_shapefile
from src.utils.points_vs_polygons import run_spatial_analysis

def main():
    cfg = Config.from_env()
    now_str = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_dir = Path(cfg.results_path) / now_str
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Configuración cargada correctamente:")
    print(cfg)

    try:
        # Obtener los registros desde diferentes fuentes
        print("\n=== OBTENIENDO LOS REGISTROS DE FIRMS ===")
        fields, rows = collect_firms_records(cfg)

        if not rows:
            raise RuntimeError("No se obtuvieron filas de FIRMS (revisa MAP_KEY, BBOX, DAYS/SOURCES).")

        # Guardar los resultados en un csv
        csv_output_path = output_dir / "firms_records.csv"
        write_csv(csv_output_path, fields, rows)

        print(f"Archivo: {csv_output_path }")
        print(f"Sources: {', '.join(cfg.sources)}")
        print(f"Días: {cfg.days}" + (f" | Fecha: {cfg.date}" if cfg.date else " | Fecha: (más reciente)"))
        print(f"BBOX: {cfg.bbox} (west,south,east,north)")
        print(f"Filas exportadas (dedup): {len(rows)}")

        # Convertir resultados a shapefile de puntos
        print("\n=== CONVERTIR CSV A SHAPEFILE DE PUNTOS ===")
        shp_output_path = convert_csv_firms_to_shapefile(csv_output_path,output_dir, cfg.points_crs)

        print("✔ Conversión completada")
        print(f"Shapefile: {shp_output_path}")

        # Correr analisis de cercanía a poligonos
        print("\n=== ANALIZAR CERCANÍA DE PUNTOS A CAPA DE POLÍGONOS ===")
        csv_computed_path = run_spatial_analysis(cfg, csv_output_path, output_dir)
        print(f"Salida: {csv_computed_path}")


        return 0

    except Exception as e:
        print(f"✖ Error: {e}", file=sys.stderr)
        return 1



if __name__ == "__main__":
    main()