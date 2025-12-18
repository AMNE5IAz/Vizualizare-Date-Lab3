from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr


@dataclass(frozen=True)
class Region:
    name: str
    # Bounding box in CDS area format: [N, W, S, E]
    area: tuple[float, float, float, float]


DEFAULT_REGIONS: list[Region] = [
    Region("Romania", (48.3, 20.2, 43.6, 29.8)),
    Region("Germany", (55.1, 5.9, 47.2, 15.2)),
    Region("Italy", (47.2, 6.6, 36.6, 18.7)),
]


def ensure_dirs() -> tuple[Path, Path]:
    data_dir = Path("data")
    out_dir = Path("outputs")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, out_dir


def download_cams_eac4(
    *,
    target_path: Path,
    variable: str,
    start_date: str,
    end_date: str,
    area: tuple[float, float, float, float],
    times: Iterable[str] = (
        "00:00",
        "03:00",
        "06:00",
        "09:00",
        "12:00",
        "15:00",
        "18:00",
        "21:00",
    ),
) -> Path:
    """
    Downloads CAMS Global Reanalysis (EAC4) gridded NetCDF via CDS API.

    Prereq: ~/.cdsapirc or %USERPROFILE%\\.cdsapirc configured.
    Docs: https://cds.climate.copernicus.eu/api-how-to
    """
    if target_path.exists():
        return target_path

    try:
        import cdsapi  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Lipsește pachetul 'cdsapi'. Instalează dependențele din requirements.txt."
        ) from exc

    if os.getenv("CAMS_SKIP_DOWNLOAD", "").strip() in {"1", "true", "True", "YES", "yes"}:
        raise SystemExit(
            f"Descărcarea este dezactivată (CAMS_SKIP_DOWNLOAD=1), dar fișierul nu există: {target_path}\n"
            "Pune manual un fișier NetCDF în folderul 'data/' sau dezactivează CAMS_SKIP_DOWNLOAD."
        )

    # CDS request dates are inclusive.
    dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()

    request = {
        "format": "netcdf",
        "variable": variable,
        "date": dates,
        "time": list(times),
        "area": list(area),
    }

    try:
        client = cdsapi.Client()
        client.retrieve("cams-global-reanalysis-eac4", request, str(target_path))
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Nu pot descărca datele CAMS prin CDS API.\n"
            "Verifică dacă ai configurat fișierul %USERPROFILE%\\.cdsapirc (URL + key), apoi reîncearcă.\n"
            "Instrucțiuni: https://cds.climate.copernicus.eu/api-how-to\n"
            f"Eroare: {exc}"
        ) from exc
    return target_path


def open_dataset(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    # Normalize coord names across NetCDF variants.
    rename = {}
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename["valid_time"] = "time"
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    return ds


def resolve_variable_name(ds: xr.Dataset, requested: str) -> str:
    """
    ADS/NetCDF exports may use short variable names (e.g. pm2p5) instead of
    the long request names used in CDS/ADS forms.
    """
    if requested in ds.data_vars:
        return requested

    aliases: dict[str, list[str]] = {
        "particulate_matter_2.5um": ["pm2p5", "pm2_5", "pm25", "pm2p5_concentration"],
        "particulate_matter_10um": ["pm10", "pm10_concentration"],
        "nitrogen_dioxide": ["no2", "nitrogen_dioxide"],
        "ozone": ["o3", "ozone"],
        "carbon_monoxide": ["co", "carbon_monoxide"],
        "sulphur_dioxide": ["so2", "sulphur_dioxide"],
    }

    for candidate in aliases.get(requested, []):
        if candidate in ds.data_vars:
            return candidate

    # Fallback: if dataset only has one variable, use it.
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))

    raise KeyError(
        f"Variabila '{requested}' nu există în fișier. Variabile: {list(ds.data_vars)}"
    )


def create_demo_dataset(path: Path, *, variable: str, start_date: str, end_date: str) -> Path:
    """
    Creates a small, synthetic gridded dataset (NetCDF) so the lab can be run
    end-to-end without CDS credentials. Not CAMS data.
    """
    rng = np.random.default_rng(7)
    times = pd.date_range(start_date, end_date, freq="3h", inclusive="both")
    lat = np.linspace(60.0, 35.0, 26)
    lon = np.linspace(-10.0, 30.0, 41)

    lats, lons = np.meshgrid(lat, lon, indexing="ij")
    base = 15 + 10 * np.exp(-((lats - 46) ** 2 + (lons - 20) ** 2) / (2 * 9**2))
    season = np.sin(np.linspace(0, 2 * np.pi, times.size))[:, None, None]
    noise = rng.normal(0, 2.5, size=(times.size, lat.size, lon.size))
    values = np.clip(base[None, :, :] + 4 * season + noise, 0, None)

    ds = xr.Dataset(
        {variable: (("time", "lat", "lon"), values.astype("float32"))},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)
    return path


def daily_mean_series(ds: xr.Dataset, var: str) -> pd.Series:
    var = resolve_variable_name(ds, var)
    da = ds[var]
    if not {"lat", "lon"} <= set(da.dims):
        raise ValueError(f"Dimensiuni neașteptate pentru {var}: {da.dims}")

    # Average spatially then aggregate daily.
    ts = da.mean(dim=("lat", "lon"), skipna=True).to_series()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()
    return ts.resample("D").mean()


def region_daily_mean(ds: xr.Dataset, var: str, region: Region) -> pd.Series:
    var = resolve_variable_name(ds, var)
    north, west, south, east = region.area
    da = ds[var].sel(lat=slice(north, south), lon=slice(west, east))
    ts = da.mean(dim=("lat", "lon"), skipna=True).to_series()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()
    return ts.resample("D").mean()


def describe_series(s: pd.Series) -> pd.Series:
    return pd.Series(
        {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)),
            "min": float(s.min()),
            "max": float(s.max()),
            "n": int(s.dropna().shape[0]),
        }
    )


def plot_time_series(daily: pd.Series, *, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    daily.plot(ax=ax, linewidth=1.5)
    daily.rolling(7, min_periods=3).mean().plot(ax=ax, linewidth=2.5, label="Media mobilă 7 zile")
    ax.set_title(title)
    ax.set_xlabel("Data")
    ax.set_ylabel("Concentrație (unități dataset)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_histogram(daily: pd.Series, *, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(daily.dropna(), bins=25, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Concentrație (unități dataset)")
    ax.set_ylabel("Frecvență")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_regions_bar(region_stats: pd.DataFrame, *, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=region_stats.reset_index().rename(columns={"index": "Regiune"}),
        x="Regiune",
        y="mean",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Media zilnică (unități dataset)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def folium_heatmap(ds: xr.Dataset, var: str, *, date: str, out_html: Path) -> None:
    import folium
    from folium.plugins import HeatMap

    day = pd.to_datetime(date).date()
    da = ds[var]
    ts = pd.to_datetime(da["time"].values)
    mask = pd.to_datetime(ts).date == day
    if not mask.any():
        raise ValueError(f"Data {date} nu există în dataset.")

    daily_grid = da.isel(time=np.where(mask)[0]).mean(dim="time", skipna=True)

    lat = daily_grid["lat"].values
    lon = daily_grid["lon"].values
    values = daily_grid.values

    lats, lons = np.meshgrid(lat, lon, indexing="ij")
    points = np.column_stack([lats.ravel(), lons.ravel(), values.ravel()])
    points = points[~np.isnan(points[:, 2])]

    center = [float(np.nanmean(lat)), float(np.nanmean(lon))]
    m = folium.Map(location=center, zoom_start=5, tiles="cartodbpositron")
    HeatMap(points.tolist(), radius=10, blur=12, max_zoom=7).add_to(m)
    m.save(str(out_html))


def main() -> None:
    data_dir, out_dir = ensure_dirs()

    variable = os.getenv("CAMS_VARIABLE", "particulate_matter_2.5um")
    start_date = os.getenv("CAMS_START", "2024-01-01")
    end_date = os.getenv("CAMS_END", "2024-01-31")

    # A bigger bounding box around Europe; used for the heatmap and overall series.
    area = (60.0, -10.0, 35.0, 30.0)

    nc_path = data_dir / f"cams_eac4_{variable}_{start_date}_{end_date}.nc"
    print(f"[1/6] Download: {nc_path}")
    try:
        if os.getenv("CAMS_DEMO", "").strip() in {"1", "true", "True", "YES", "yes"}:
            print("CAMS_DEMO=1: generez dataset demo (sintetic, nu CAMS).")
            create_demo_dataset(nc_path, variable=variable, start_date=start_date, end_date=end_date)
        else:
            download_cams_eac4(
                target_path=nc_path,
                variable=variable,
                start_date=start_date,
                end_date=end_date,
                area=area,
            )
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Eroare neașteptată la pregătirea datelor: {exc}") from exc

    print("[2/6] Load NetCDF")
    ds = open_dataset(nc_path)
    resolved_var = resolve_variable_name(ds, variable)
    if resolved_var != variable:
        print(f"Nota: variabila ceruta '{variable}' apare in fisier ca '{resolved_var}'.")
        variable = resolved_var

    print("[3/6] Preprocess + daily means (Europe)")
    europe_daily = daily_mean_series(ds, variable)
    (out_dir / "daily_series.csv").write_text(europe_daily.to_csv(), encoding="utf-8")

    stats = describe_series(europe_daily)
    stats.to_frame("Europe").T.to_csv(out_dir / "stats_europe.csv", index=False)

    print("[4/6] Regions comparison")
    region_daily = {r.name: region_daily_mean(ds, variable, r) for r in DEFAULT_REGIONS}
    region_stats = pd.DataFrame({name: describe_series(s) for name, s in region_daily.items()}).T
    region_stats.to_csv(out_dir / "stats_regions.csv", index=True)

    print("[5/6] Plots")
    plot_time_series(
        europe_daily,
        title=f"CAMS EAC4 {variable} - media zilnică (Europa)\n{start_date} → {end_date}",
        out_path=out_dir / "timeseries_europe.png",
    )
    plot_histogram(
        europe_daily,
        title=f"Distribuția mediilor zilnice - {variable} (Europa)",
        out_path=out_dir / "hist_europe.png",
    )
    plot_regions_bar(
        region_stats,
        title=f"Comparație regiuni - media zilnică ({variable})",
        out_path=out_dir / "regions_bar.png",
    )

    print("[6/6] Heatmap map (Folium)")
    heat_date = os.getenv("CAMS_HEAT_DATE", start_date)
    folium_heatmap(ds, variable, date=heat_date, out_html=out_dir / f"heatmap_{heat_date}.html")

    print("Gata. Vezi folderul 'outputs/'.")


if __name__ == "__main__":
    main()
