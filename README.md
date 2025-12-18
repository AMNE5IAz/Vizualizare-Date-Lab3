# Vizualizare Date — Laborator 3 (CAMS)

## Rulare rapidă
1) Instalează dependențele:
`pip install -r requirements.txt`

2) Configurează CDS API (necesar pentru descărcarea datelor CAMS):
- Creează fișierul `%USERPROFILE%\\.cdsapirc` conform instrucțiunilor: https://cds.climate.copernicus.eu/api-how-to

3) Rulează analiza:
`python lab3_cams.py`

## Configurări (opțional)
Scriptul citește variabile de mediu:
- `CAMS_VARIABLE` (implicit `particulate_matter_2.5um`)
- `CAMS_START` (implicit `2024-01-01`)
- `CAMS_END` (implicit `2024-01-31`)
- `CAMS_HEAT_DATE` (implicit `CAMS_START`)

Exemplu:
`$env:CAMS_VARIABLE='nitrogen_dioxide'; $env:CAMS_START='2024-02-01'; $env:CAMS_END='2024-02-29'; python lab3_cams.py`

## Ieșiri
Toate fișierele generate sunt în `outputs/`.

