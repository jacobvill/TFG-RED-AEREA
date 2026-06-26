# Simulación y Análisis del Impacto Operativo de la Red Aérea Global

Trabajo de Fin de Grado · Ingeniería Informática · Universidad Alfonso X el Sabio (UAX)
Autor: **Jacob Altenburger Villar** — 2026

Aplicación web que modela la red de transporte aéreo como un grafo y simula qué ocurre
cuando un aeropuerto crítico falla: cómo se redistribuyen los vuelos (efecto cascada) y
qué coste tiene en tiempo, combustible y emisiones de CO2.

## Funcionalidades

- **Tráfico en tiempo real:** mapa interactivo de vuelos a partir de datos ADS-B (OpenSky).
- **Simulador de crisis:** cierre o reducción de capacidad de un aeropuerto y propagación
  del impacto por la red mediante un algoritmo de cascada multinivel (BFS).
- **Análisis de red:** métricas de centralidad (betweenness, grado) y aislamiento geográfico
  para identificar los aeropuertos más críticos.

## Requisitos

- Python 3.10 o superior
- Las librerías del archivo `requirements.txt`:

```
streamlit
plotly
pandas
numpy
networkx
scikit-learn
openpyxl
trino
```

Instalación:

```bash
pip install -r requirements.txt
```

## Ejecución

Desde la carpeta del proyecto:

```bash
streamlit run Inicio.py
```

Se abrirá en el navegador. Usa la barra lateral para moverte entre las páginas.

> Para el análisis histórico (Trino) HACE FALTA UNA CUENTA CREADA EN OPENSKY.
> La visualización en tiempo real usa la API REST de OpenSky con credenciales OAuth2.

## Estructura

```
.
├── Inicio.py                  # Portada y punto de entrada (streamlit run Inicio.py)
├── pages/                     # Páginas de la app
│   ├── 1_Proyecto.py          # Tráfico en tiempo real
│   ├── 2_Simulador.py         # Simulador de crisis
│   └── 3_Analisis_Red.py      # Análisis de red
├── airports.csv               # Base de aeropuertos (OurAirports)
├── capacidad_aterrizaje_aeropuertos_europa.xlsx
├── parking_aeropuertos.xlsx
└── requirements.txt
```

## Fuentes de datos

- **OpenSky Network** (API REST y motor Trino) — posiciones y vuelos históricos
- **adsbdb.com** — modelo de aeronave y ruta por vuelo
- **OurAirports** — base de datos de aeropuertos y pistas
- **AENA / DGAC y Eurocontrol** — capacidades aeroportuarias
