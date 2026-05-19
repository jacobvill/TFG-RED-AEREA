from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from datetime import datetime, timezone

user = "jaltevil@myuax.com"

conn = connect(
    host="trino.opensky-network.org",
    port=443,
    user=user,
    auth=OAuth2Authentication(),
    http_scheme="https",
    catalog="minio",
    schema="osky",
    request_timeout=120.0
)

dt_utc = datetime(2026, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
ts_day = int(dt_utc.timestamp())

print("=" * 60)
print(f"Analizando flights_data4 para el día {dt_utc.date()}")
print("=" * 60)

cur = conn.cursor()

# 1. Total de vuelos ese día
cur.execute(f"SELECT COUNT(*) FROM flights_data4 WHERE day = {ts_day}")
total = cur.fetchone()[0]
print(f"\nTotal vuelos ese día: {total:,}")

# 2. Nulls en estarrivalairport
cur.execute(f"""
    SELECT
        COUNT(*) AS total,
        COUNT(estarrivalairport) AS con_destino,
        COUNT(*) - COUNT(estarrivalairport) AS sin_destino
    FROM flights_data4
    WHERE day = {ts_day}
""")
row = cur.fetchone()
print(f"\nestarrivalairport:")
print(f"  Con destino:   {row[1]:,} ({row[1]/row[0]*100:.1f}%)")
print(f"  Sin destino:   {row[2]:,} ({row[2]/row[0]*100:.1f}%)")

# 3. Nulls en estdepartureairport
cur.execute(f"""
    SELECT
        COUNT(*) AS total,
        COUNT(estdepartureairport) AS con_origen,
        COUNT(*) - COUNT(estdepartureairport) AS sin_origen
    FROM flights_data4
    WHERE day = {ts_day}
""")
row = cur.fetchone()
print(f"\nestdepartureairport:")
print(f"  Con origen:    {row[1]:,} ({row[1]/row[0]*100:.1f}%)")
print(f"  Sin origen:    {row[2]:,} ({row[2]/row[0]*100:.1f}%)")

# 4. Vuelos con ambos campos rellenos
cur.execute(f"""
    SELECT COUNT(*)
    FROM flights_data4
    WHERE day = {ts_day}
      AND estdepartureairport IS NOT NULL
      AND estarrivalairport   IS NOT NULL
""")
ambos = cur.fetchone()[0]
print(f"\nVuelos con origen Y destino conocidos: {ambos:,} ({ambos/total*100:.1f}%)")

# 5. Top 10 aeropuertos de destino más frecuentes
print("\nTop 10 destinos más frecuentes ese día:")
cur.execute(f"""
    SELECT estarrivalairport, COUNT(*) AS cnt
    FROM flights_data4
    WHERE day = {ts_day}
      AND estarrivalairport IS NOT NULL
    GROUP BY estarrivalairport
    ORDER BY cnt DESC
    LIMIT 10
""")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]:,} vuelos")

# 6. Top 10 aeropuertos de origen más frecuentes
print("\nTop 10 orígenes más frecuentes ese día:")
cur.execute(f"""
    SELECT estdepartureairport, COUNT(*) AS cnt
    FROM flights_data4
    WHERE day = {ts_day}
      AND estdepartureairport IS NOT NULL
    GROUP BY estdepartureairport
    ORDER BY cnt DESC
    LIMIT 10
""")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]:,} vuelos")