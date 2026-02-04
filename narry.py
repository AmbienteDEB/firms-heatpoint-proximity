import os
import requests
import pandas as pd
import geopandas as gpd
import arcpy

# --- ENTRADA DE USUARIO ---
# Pais ingresado entre corchetes
pais_input = "[el salvador]"
pais_limpio = pais_input.strip("[]")

# Configuración de API y Rutas
api_key = "4c8d1a8caf13fcdc06d6b9f2288e2bb8"
folder_path = r"C:\NASA_FIRMS_Fires"
shp_name = f"Fires_{pais_limpio.replace(' ', '_')}.shp"
full_path = os.path.join(folder_path, shp_name)

print("--- INICIANDO PROCESO NASA FIRMS ---")

# 1. Crear carpeta local si no existe
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"✅ Carpeta creada: {folder_path}")
else:
    print(f"✅ Carpeta verificada: {folder_path}")

# 2. Obtener centro geográfico vía REST Countries
print(f"🌍 Consultando centro geográfico para: {pais_limpio}...")
try:
    # Usamos el endpoint de nombre completo para mayor precisión
    url_pais = f"https://restcountries.com/v3.1/name/{pais_limpio.replace(' ', '%20')}?fullText=true"
    r = requests.get(url_pais)
    r.raise_for_status()
    datos = r.json()[0]

    # latlng viene como [latitud, longitud]
    lat_c = datos['latlng'][0]
    lon_c = datos['latlng'][1]
    print(f"📍 Centro obtenido: Lat {lat_c}, Lon {lon_c}")

except Exception as e:
    print(f"❌ Error al obtener coordenadas del país: {e}")
    raise

# 3. Crear Bounding Box expandido (+/- 5 grados)
# Formato requerido por FIRMS: min_lon, min_lat, max_lon, max_lat
west = lon_c - 5
south = lat_c - 5
east = lon_c + 5
north = lat_c + 5
bbox_str = f"{west},{south},{east},{north}"
print(f"📐 Bounding Box (±5°): {bbox_str}")

# 4. Construir URL y descargar CSV de FIRMS
url_firms = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/{bbox_str}/7"
print("🛰️ Descargando datos desde NASA FIRMS (últimos 7 días)...")

try:
    # Leemos directamente el CSV desde la URL usando pandas
    df = pd.read_csv(url_firms)
    print(f"📥 Datos descargados. Filas iniciales: {len(df)}")

    # 5. Filtrar únicamente confidence == 'n'
    # En VIIRS, 'n' representa confianza nominal
    df_filtered = df[df['confidence'] == 'n'].copy()
    print(f"🔥 Registros tras filtrar confianza 'n': {len(df_filtered)}")

    if df_filtered.empty:
        print("⚠️ No hay datos para los criterios seleccionados. Proceso finalizado.")
    else:
        # 6. Generar GeoDataFrame y exportar a Shapefile (EPSG:4326)
        print("🛠️ Generando geometría con GeoPandas...")
        gdf = gpd.GeoDataFrame(
            df_filtered,
            geometry=gpd.points_from_xy(df_filtered.longitude, df_filtered.latitude),
            crs="EPSG:4326"
        )

        # Eliminar si ya existe para evitar errores de sobreescritura
        if os.path.exists(full_path):
            arcpy.management.Delete(full_path)

        gdf.to_file(full_path)
        print(f"💾 Shapefile exportado exitosamente en: {full_path}")

        # 7. Agregar al mapa activo de ArcGIS Pro
        print("🗺️ Agregando capa al mapa activo...")
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        mapa = aprx.activeMap

        if mapa is not None:
            mapa.addDataFromPath(full_path)
            print("🚀 ¡Capa agregada correctamente al mapa!")
        else:
            print("❌ No se encontró un mapa activo. El archivo se guardó pero no se visualizó.")

except Exception as e:
    print(f"❌ Error durante el proceso: {e}")

print("--- FIN DEL SCRIPT ---")