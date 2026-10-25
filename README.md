# V3.5 preprocessed roads

1. Install dependencies locally with `..\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
2. Run from this directory:

```powershell
..\.venv\Scripts\python.exe preprocess_ma_roads.py --pbf "..\osm_data\massachusetts-latest.osm.pbf" --excel "..\Intersections.xlsx" --output "data\ma_six_city_roads.json.gz"
```

3. Confirm the resulting gzip file exists. Run `..\.venv\Scripts\python.exe -m streamlit run app.py`.
4. For cloud hosting, deploy `app.py`, `requirements.txt`, and `data/ma_six_city_roads.json.gz`. Do not upload your `.env` or API key; configure host secrets instead.

This extracts roads named in the workbook (statewide matching), not a geographic municipal clip. It works for intersections listed in that workbook; adding roads requires rerunning the preprocessor. The app still uses geocoding and local distance filters to select the actual intersection.
