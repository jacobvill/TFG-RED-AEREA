"""
comprobar_destinos.py
Comprueba en Trino que fraccion REAL de vuelos no tiene origen/destino verificado,
para decidir si el "factor de escala del 24%" tiene sentido o no.

Ejecutar:  python comprobar_destinos.py
(Requiere tus credenciales; abrira el navegador la primera vez para autenticar.)
"""
from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from datetime import datetime, timezone

USUARIO = "jaltevil@myuax.com"
FECHA   = datetime(2024, 1, 16)        # <- cambia el dia que quieras comprobar
ICAO    = "LEMD"                        # <- aeropuerto a inspeccionar

ts_day = int(datetime(FECHA.year, FECHA.month, FECHA.day, 0, 0, 0, tzinfo=timezone.utc).timestamp())

conn = connect(host="trino.opensky-network.org", port=443, user=USUARIO,
               auth=OAuth2Authentication(), http_scheme="https",
               catalog="minio", schema="osky", request_timeout=120.0)
cur = conn.cursor()

print(f"Dia: {FECHA.date()}  ·  aeropuerto: {ICAO}\n")

# 1) % real de vuelos sin destino y sin origen en TODA la red ese dia
cur.execute(f"""
    SELECT COUNT(*)                      AS total,
           COUNT(estarrivalairport)      AS con_destino,
           COUNT(estdepartureairport)    AS con_origen
    FROM flights_data4
    WHERE day = {ts_day}
""")
total, con_dest, con_orig = cur.fetchone()
print("=== Toda la red ===")
print(f"  Total vuelos:   {total:,}")
print(f"  Sin destino:    {total - con_dest:,}  ({100*(total-con_dest)/total:.1f} %)")
print(f"  Sin origen:     {total - con_orig:,}  ({100*(total-con_orig)/total:.1f} %)")

# 2) Llegadas y salidas CONFIRMADAS de ese aeropuerto (deberian estar equilibradas
#    en un hub; si las llegadas son muchas menos, es senal de subconteo)
cur.execute(f"""
    SELECT
      (SELECT COUNT(*) FROM flights_data4 WHERE day={ts_day} AND estarrivalairport='{ICAO}')   AS llegadas,
      (SELECT COUNT(*) FROM flights_data4 WHERE day={ts_day} AND estdepartureairport='{ICAO}')  AS salidas
""")
lleg, sal = cur.fetchone()
print(f"\n=== {ICAO} (confirmados) ===")
print(f"  Llegadas con destino {ICAO}: {lleg:,}")
print(f"  Salidas  con origen  {ICAO}: {sal:,}")
print(f"  (en un hub deberian ser parecidas; gran descuadre = subconteo)")

print("\nListo. Dime que numeros salen y decidimos que hacemos con el factor de escala.")