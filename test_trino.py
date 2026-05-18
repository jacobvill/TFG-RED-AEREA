from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from datetime import datetime

user = "jaltevil@myuax.com"   # en minúsculas

conn = connect(
    host="trino.opensky-network.org",
    port=443,
    user=user,
    auth=OAuth2Authentication(),
    http_scheme="https",
    catalog="minio",
    schema="osky",
    request_timeout=60.0
)

# Hora de prueba: 1 de enero 2024 a las 12:00 UTC
ts_hour = int(datetime(2024, 1, 1, 12, 0, 0).timestamp())

query = f"""
    SELECT icao24, callsign, lat, lon
    FROM state_vectors_data4
    WHERE hour = {ts_hour}
      AND time BETWEEN {ts_hour} AND {ts_hour} + 30
      AND onground = false
      AND lat IS NOT NULL
    LIMIT 10
"""

cur = conn.cursor()
cur.execute(query)
rows = cur.fetchall()
print(f"Filas recibidas: {len(rows)}")
for r in rows:
    print(r)