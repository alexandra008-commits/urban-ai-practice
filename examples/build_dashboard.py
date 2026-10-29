"""Собирает dashboard.html из готовых результатов ноутбука vyborg_blocksnet_study.ipynb.

Скрипт только читает файлы из examples/data/vyborg и ничего не скачивает и не
пересчитывает. Запуск из папки проекта с активированным окружением .venv:

    python examples/build_dashboard.py

Результат: examples/dashboard.html. Для подложки карты и библиотек Leaflet и
Chart.js при открытии страницы нужен интернет.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping
from shapely.ops import transform


def find_project_root(start: Path) -> Path:
    """Ищет папку, в которой лежит pyproject.toml."""
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
DATA_DIR = PROJECT_ROOT / "examples" / "data" / "vyborg"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
OUTPUT_PATH = PROJECT_ROOT / "examples" / "dashboard.html"

TERRITORY_LABEL = "Выборг"
SIMPLIFY_TOLERANCE_M = 1.5  # упрощение контуров только для веса страницы, на значения не влияет

# Описание всех показателей: колонка, название, группа, единицы, тип шкалы, источник, смысл.
# kind: seq — числовая шкала, div — доля 0…1 (дефицит ↔ норма), cat — категории.
METRICS = [
    ("land_use", "Преобладающая функция", "Землепользование", "категория", "cat", "OSM landuse + правила pipeline",
     "Функция с наибольшей долей площади квартала. Выбирается даже при доле 1–2 %."),
    ("share", "Доля преобладающей функции", "Землепользование", "доля 0…1", "seq", "расчёт BlocksNet",
     "Какую часть квартала занимает преобладающая функция. Ниже 0,2 — тип квартала ненадёжен."),
    ("residential", "Доля жилой зоны", "Землепользование", "доля 0…1", "seq", "OSM landuse", "Часть площади квартала в жилой зоне."),
    ("business", "Доля деловой зоны", "Землепользование", "доля 0…1", "seq", "OSM landuse", "Часть площади в торговой и деловой зоне."),
    ("recreation", "Доля рекреации", "Землепользование", "доля 0…1", "seq", "OSM landuse", "Часть площади в лесу, парках, газонах."),
    ("industrial", "Доля промзоны", "Землепользование", "доля 0…1", "seq", "OSM landuse", "Часть площади в промышленной зоне и гаражах."),
    ("transport", "Доля транспортной зоны", "Землепользование", "доля 0…1", "seq", "OSM landuse", "Часть площади в транспортной зоне."),
    ("special", "Доля спецзоны", "Землепользование", "доля 0…1", "seq", "OSM landuse", "Кладбища, стройки, военные и религиозные территории."),
    ("agriculture", "Доля сельхозземель", "Землепользование", "доля 0…1", "seq", "OSM landuse", "Часть площади в сельскохозяйственной зоне."),
    ("site_area_ha", "Площадь квартала", "Землепользование", "га", "seq", "расчёт по контуру", "Площадь квартала между улицами, железной дорогой и водой."),
    ("median_time_min", "Медианное время до кварталов", "Доступность", "минуты", "seq", "расчёт по пешеходному графу",
     "Типичное время пешком (5 км/ч) от квартала до остальных. Меньше — квартал ближе ко всему городу."),
    ("area_accessibility", "Area accessibility", "Доступность", "минуты", "seq", "расчёт BlocksNet",
     "Среднее время пешком до остальных кварталов с весом по их площади. Меньше — лучше."),
    ("count", "Число зданий", "Здания и население", "шт.", "seq", "OSM", "Сколько зданий стоит в квартале."),
    ("is_living", "Жилых зданий", "Здания и население", "шт.", "seq", "OSM (тег building)", "Сколько зданий считаются жилыми."),
    ("floors_from_osm", "Зданий с этажностью из OSM", "Здания и население", "шт.", "seq", "OSM",
     "У скольких зданий этажность указана в OSM. У остальных подставлен 1 этаж."),
    ("footprint_area", "Пятно застройки", "Здания и население", "м²", "seq", "OSM", "Площадь, которую здания занимают на земле."),
    ("build_floor_area", "Площадь этажей", "Здания и население", "м²", "seq", "оценка", "Пятно × этажность; без этажности — 1 этаж."),
    ("living_area", "Жилая площадь", "Здания и население", "м²", "seq", "оценка", "90 % площади этажей жилых зданий."),
    ("non_living_area", "Нежилая площадь", "Здания и население", "м²", "seq", "оценка", "Площадь этажей минус жилая."),
    ("population", "Население", "Здания и население", "человек", "seq", "оценка", "Жилая площадь ÷ 30 м² на человека, минимум 1 на жилой дом."),
    ("schools_count", "Школ в квартале", "Школы", "шт.", "seq", "OSM (amenity=school)", "Сколько школ стоит в квартале."),
    ("capacity", "Мест в школах", "Школы", "места", "seq", "OSM + оценка", "Суммарная вместимость школ квартала."),
    ("capacity_osm", "Места из OSM", "Школы", "места", "seq", "OSM", "Вместимость, указанная в OSM тегом capacity."),
    ("capacity_estimated", "Подставленные места", "Школы", "места", "seq", "оценка",
     "Вместимость, которую подставил BlocksNet (250 мест), когда в OSM её нет."),
    ("capacity_imputed", "Школ с подставленной вместимостью", "Школы", "шт.", "seq", "оценка", "Сколько школ квартала получили типовую вместимость."),
    ("demand", "Потребность в местах", "Обеспеченность", "места", "seq", "оценка", "Население ÷ 1000 × 120 мест, с округлением вверх."),
    ("demand_within", "Закрыто в пределах 15 мин", "Обеспеченность", "места", "seq", "расчёт BlocksNet",
     "Сколько мест жители квартала получили в школах не дальше 15 минут пешком."),
    ("demand_without", "Закрыто дальше 15 мин", "Обеспеченность", "места", "seq", "расчёт BlocksNet",
     "Места в дальних школах. При max_depth = 1 дальние школы не ищутся, поэтому здесь нули."),
    ("demand_left", "Не хватает мест", "Обеспеченность", "места", "seq", "расчёт BlocksNet", "Сколько мест жителям квартала не нашлось."),
    ("capacity_within", "Отдано мест рядом", "Обеспеченность", "места", "seq", "расчёт BlocksNet", "Сколько мест школы квартала отдали жителям в пределах 15 минут."),
    ("capacity_without", "Отдано мест далеко", "Обеспеченность", "места", "seq", "расчёт BlocksNet", "Сколько мест отдано жителям дальше 15 минут."),
    ("capacity_left", "Свободные места", "Обеспеченность", "места", "seq", "расчёт BlocksNet", "Сколько мест в школах квартала осталось незанятыми."),
    ("provision_strong", "Обеспеченность strong", "Обеспеченность", "доля 0…1", "div", "расчёт BlocksNet",
     "Доля потребности, закрытая школами в пределах 15 минут. 1 — мест хватает, 0 — нет."),
    ("provision_weak", "Обеспеченность weak", "Обеспеченность", "доля 0…1", "div", "расчёт BlocksNet",
     "То же с учётом дальних школ. Сейчас совпадает со strong (max_depth = 1)."),
    ("fsi", "FSI — интенсивность", "Плотность", "м²/м²", "seq", "расчёт (оценка этажей)", "Площадь этажей ÷ площадь квартала."),
    ("gsi", "GSI — застроенность", "Плотность", "доля 0…1", "seq", "расчёт по OSM", "Пятно застройки ÷ площадь квартала."),
    ("mxi", "MXI — доля жилья", "Плотность", "доля 0…1", "seq", "расчёт (оценка)", "Жилая площадь ÷ площадь этажей."),
    ("l", "L — средняя этажность", "Плотность", "этажи", "seq", "расчёт (оценка этажей)", "Площадь этажей ÷ пятно застройки, не меньше 1."),
    ("osr", "OSR — открытое пространство", "Плотность", "м²/м²", "seq", "расчёт", "Незастроенная земля на 1 м² площади этажей."),
    ("share_living", "Жилая площадь на пятно", "Плотность", "м²/м²", "seq", "расчёт (оценка)", "Жилая площадь ÷ пятно застройки, может быть больше 1."),
    ("share_non_living", "Нежилая площадь на пятно", "Плотность", "м²/м²", "seq", "расчёт (оценка)", "Нежилая площадь ÷ пятно застройки, может быть больше 1."),
    ("morphotype_strelka", "Морфотип Strelka", "Типы застройки", "категория", "cat", "правила BlocksNet",
     "Тип застройки по фиксированным порогам этажности, плотности и доли жилья."),
    ("morphotype_spacematrix", "Морфотип Spacematrix", "Типы застройки", "категория", "cat", "кластеризация KMeans",
     "Группа похожих кварталов этого города по l, fsi, mxi."),
]
# Колонки без смысла на карте: остаются только в таблице.
TECHNICAL_COLUMNS = {"number_of_floors": "Сумма этажей всех зданий (не средняя!)", "cluster": "Номер кластера Spacematrix"}

LAND_USE_LABELS = {
    "residential": "жилая", "business": "деловая", "recreation": "рекреация", "industrial": "промышленная",
    "transport": "транспорт", "special": "специальная", "agriculture": "сельхоз", "no_data": "нет данных",
}
# Порядок слотов категориальной палитры подобран по смыслу: рекреация — зелёный слот и т.д.
LAND_USE_SLOTS = {
    "transport": 1, "residential": 2, "agriculture": 3, "special": 4, "business": 5, "recreation": 6, "industrial": 7,
}


def read_layer(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Нет файла {path}. Сначала выполните ноутбук до конца.")
    return gpd.read_file(path)


def clean(value, digits=4):
    """Превращает значение в то, что можно записать в JSON: без NaN, inf и numpy-типов."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        rounded = round(float(value), digits)
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def records(df: pd.DataFrame, digits=3) -> list[dict]:
    return [{str(k): clean(v, digits) for k, v in row.items()} for row in df.to_dict(orient="records")]


def round_geometry(geom, digits=6):
    return transform(lambda x, y, z=None: (np.round(x, digits), np.round(y, digits)), geom)


def load_data():
    results = read_layer(RESULTS_DIR / "blocks_results.geojson")
    boundary = read_layer(RAW_DIR / "boundary.geojson")
    schools = read_layer(PROCESSED_DIR / "schools_imputed.geojson")
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    links = pd.read_csv(RESULTS_DIR / "provision_links.csv")
    clusters = pd.read_csv(RESULTS_DIR / "spacematrix_clusters.csv")
    return results, boundary, schools, summary, links, clusters


def prepare_blocks(results: gpd.GeoDataFrame, schools: gpd.GeoDataFrame):
    metric_crs = results.estimate_utm_crs()
    blocks = results.to_crs(metric_crs).copy()
    blocks["block_id"] = blocks["block_id"].astype(int)
    blocks.index = blocks["block_id"].values
    blocks["site_area_ha"] = blocks["site_area"] / 10_000

    # Число школ в квартале: в итоговом слое колонка count относится к зданиям.
    school_points = schools.to_crs(metric_crs).copy()
    school_points["geometry"] = school_points.representative_point()
    block_shapes = gpd.GeoDataFrame({"bid": blocks.index.values}, geometry=blocks.geometry.values, crs=metric_crs)
    joined = school_points.sjoin(block_shapes, how="inner", predicate="within")
    blocks["schools_count"] = joined.groupby("bid").size().reindex(blocks.index).fillna(0)
    school_points["block_id"] = joined["bid"].reindex(school_points.index)

    for column in ["land_use", "morphotype_strelka", "morphotype_spacematrix"]:
        blocks[column] = blocks[column].fillna("no_data")

    simplified = blocks.geometry.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
    simplified = simplified.where(~simplified.is_empty, blocks.geometry)
    blocks_wgs = gpd.GeoDataFrame(blocks.drop(columns="geometry"), geometry=simplified, crs=metric_crs).to_crs(4326)
    return blocks, blocks_wgs, school_points.to_crs(4326), metric_crs


def build_geojson(blocks_wgs: gpd.GeoDataFrame, columns: list[str]) -> dict:
    features = []
    for block_id, row in blocks_wgs.iterrows():
        properties = {"block_id": int(block_id)}
        for column in columns:
            if column in row:
                properties[column] = clean(row[column], 4)
        features.append({"type": "Feature", "properties": properties, "geometry": mapping(round_geometry(row.geometry))})
    return {"type": "FeatureCollection", "features": features}


def school_features(school_points: gpd.GeoDataFrame) -> list[dict]:
    items = []
    for _, row in school_points.iterrows():
        imputed = row.get("capacity_imputed")
        if isinstance(imputed, str):
            imputed = imputed.strip().lower() in {"true", "1"}
        items.append({
            "block_id": clean(row.get("block_id")),
            "lat": round(row.geometry.y, 6),
            "lon": round(row.geometry.x, 6),
            "capacity": clean(row.get("capacity")),
            "imputed": bool(imputed),
        })
    return items


def category_order(series: pd.Series, fixed_slots: dict | None = None) -> list[dict]:
    """Категории с номером слота палитры. Слоты по порядку, «нет данных» — нейтральный серый."""
    counts = series.value_counts()
    categories = []
    if fixed_slots:
        for name, slot in sorted(fixed_slots.items(), key=lambda item: item[1]):
            if name in counts.index:
                categories.append({"value": name, "label": LAND_USE_LABELS.get(name, name), "slot": slot})
    else:
        named = [name for name in counts.index if name != "no_data"]
        for position, name in enumerate(named):
            categories.append({"value": name, "label": name, "slot": position + 1 if position < 8 else 0})
    if "no_data" in counts.index:
        categories.append({"value": "no_data", "label": "нет данных", "slot": 0})
    return categories


def fmt(number: float) -> str:
    """Целое число с пробелом между разрядами: 71 811."""
    return f"{number:,.0f}".replace(",", " ")


def build_conclusions(blocks: gpd.GeoDataFrame, summary: dict) -> list[dict]:
    """Короткие выводы с цифрами. kind: finding — вывод, caveat — ограничение."""
    items = []
    population = float(blocks["population"].sum())
    demand = float(blocks["demand"].sum())
    within = float(blocks["demand_within"].sum())
    left = float(blocks["demand_left"].sum())
    capacity = float(blocks["capacity"].sum())
    schools_total = int(blocks["schools_count"].sum())
    imputed = int(blocks["capacity_imputed"].sum())
    populated = blocks[blocks["population"] > 0]
    zero_provision = populated[populated["provision_strong"].fillna(0) <= 0.001]

    items.append({
        "kind": "finding",
        "title": "Школьных мест по модели меньше половины от нужного",
        "text": (
            f"На оценочные {fmt(population)} жителей по нормативу нужно {fmt(demand)} мест, "
            f"а в {schools_total} школах {fmt(capacity)} мест. В пределах 15 минут пешком закрыто "
            f"{fmt(within)} мест ({within / demand:.0%} потребности), не хватает {fmt(left)}. "
            f"{len(zero_provision)} жилых кварталов с {fmt(zero_provision['population'].sum())} жителями "
            "не получили ни одного места рядом."
        ),
    })
    if imputed == schools_total and schools_total > 0:
        items.append({
            "kind": "caveat",
            "title": "Вместимость всех школ подставлена",
            "text": (
                f"Ни у одной из {schools_total} школ в OSM нет тега capacity, поэтому каждой подставлено "
                "250 мест — минимальный типовой размер из справочника BlocksNet. Реальные школы обычно "
                "больше, так что дефицит на карте, скорее всего, преувеличен. Для выводов нужны реальные "
                "данные о вместимости."
            ),
        })
    elif imputed > 0:
        items.append({
            "kind": "caveat",
            "title": "Часть вместимости школ подставлена",
            "text": f"У {imputed} из {schools_total} школ вместимость не указана в OSM и заменена на 250 мест.",
        })

    floors_share = float(summary.get("buildings_with_osm_floors_share", np.nan))
    strelka = blocks.loc[blocks["morphotype_strelka"] != "no_data", "morphotype_strelka"]
    low_rise = strelka.isin(["low-rise non-residential", "low-rise model", "individual residential"]).mean() if len(strelka) else np.nan
    items.append({
        "kind": "caveat",
        "title": "Этажность почти везде неизвестна",
        "text": (
            f"Этажность указана в OSM только у {floors_share:.0%} зданий, остальные считаются одноэтажными. "
            f"Поэтому {low_rise:.0%} застроенных кварталов попали в малоэтажные типы Strelka, а площадь "
            "этажей, плотность FSI и население многоэтажек занижены."
        ),
    })

    top_types = strelka.value_counts().head(3)
    if len(top_types):
        listed = ", ".join(f"{name} — {count}" for name, count in top_types.items())
        items.append({
            "kind": "finding",
            "title": "Преобладающие типы застройки",
            "text": f"Чаще всего встречаются морфотипы Strelka: {listed} кварталов. Кварталы без зданий тип не получают.",
        })

    dense = blocks[blocks["fsi"] > 1].sort_values("fsi", ascending=False)
    if len(dense):
        ids = ", ".join(str(i) for i in dense.index[:5])
        items.append({
            "kind": "finding",
            "title": "Плотная застройка сосредоточена в немногих кварталах",
            "text": (
                f"Только у {len(dense)} кварталов FSI больше 1, то есть площадь этажей больше площади земли. "
                f"Самые плотные: {ids}. Найдите их на карте слоем «FSI»."
            ),
        })

    median_time = float(summary.get("median_travel_time_min", blocks["median_time_min"].median()))
    best = blocks.nsmallest(5, "area_accessibility")
    items.append({
        "kind": "finding",
        "title": "Пешком город большой",
        "text": (
            f"Медианное время пешком между двумя кварталами — {median_time:.0f} мин: в границы города входят "
            f"большие леса и заливы. Лучшая area accessibility ({best['area_accessibility'].min():.0f} мин) "
            f"у кварталов {', '.join(str(i) for i in best.index)}."
        ),
    })

    land = blocks[blocks["land_use"] != "no_data"]
    weak = land[land["share"] < 0.2]
    no_data_count = int((blocks["land_use"] == "no_data").sum())
    items.append({
        "kind": "caveat",
        "title": "Функция квартала часто определена ненадёжно",
        "text": (
            f"У {no_data_count} из {len(blocks)} кварталов нет данных landuse в OSM. Ещё у {len(weak)} "
            f"кварталов ({fmt(weak['site_area_ha'].sum())} га) преобладающая функция занимает меньше 20 % "
            "площади: большой лесной квартал может получить «жилая» по маленькому участку."
        ),
    })

    if (blocks["demand_without"].sum() == 0) and np.allclose(
        blocks["provision_strong"].fillna(-1), blocks["provision_weak"].fillna(-1)
    ):
        items.append({
            "kind": "caveat",
            "title": "provision_weak совпадает с provision_strong",
            "text": (
                "Расчёт выполнен как в pipeline.ipynb с max_depth = 1: школы дальше 15 минут не "
                "рассматриваются. Чтобы увидеть дальние школы, в ноутбуке нужно поставить "
                "PROVISION_MAX_DEPTH = 2 и пересчитать раздел 9."
            ),
        })

    items.append({
        "kind": "caveat",
        "title": "Это модель на открытых данных",
        "text": (
            "Все показатели построены по OpenStreetMap: население — оценка по площади зданий, граф — "
            "пешеходные дороги OSM со скоростью 5 км/ч без учёта закрытых территорий. Значения подходят "
            "для сравнения кварталов между собой, но не заменяют официальную статистику."
        ),
    })
    return items


def build_payload():
    results, boundary, schools, summary_df, links, clusters = load_data()
    blocks, blocks_wgs, school_points, metric_crs = prepare_blocks(results, schools)
    summary = dict(zip(summary_df["metric"], summary_df["value"]))
    for key, value in summary.items():
        try:
            summary[key] = float(value)
        except (TypeError, ValueError):
            pass

    metric_columns = [m[0] for m in METRICS if m[0] in blocks.columns]
    table_columns = metric_columns + [c for c in TECHNICAL_COLUMNS if c in blocks.columns]
    geojson = build_geojson(blocks_wgs, table_columns)

    metrics = []
    for column, label, group, unit, kind, source, description in METRICS:
        if column not in blocks.columns:
            continue
        item = {"key": column, "label": label, "group": group, "unit": unit, "kind": kind,
                "source": source, "description": description}
        if kind == "cat":
            item["categories"] = category_order(blocks[column], LAND_USE_SLOTS if column == "land_use" else None)
        metrics.append(item)
    for column, label in TECHNICAL_COLUMNS.items():
        if column in blocks.columns:
            metrics.append({"key": column, "label": label, "group": "Технические", "unit": "—", "kind": "tech",
                            "source": "расчёт BlocksNet", "description": label})

    land_area = (
        blocks.groupby("land_use")["site_area_ha"].sum().reindex(
            [c["value"] for c in category_order(blocks["land_use"], LAND_USE_SLOTS)]
        )
    )

    # Готовая матрица доступности: только читается, не пересчитывается.
    matrix_path = RESULTS_DIR / "accessibility_matrix.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Нет файла {matrix_path}. Сначала выполните ноутбук до конца.")
    matrix = pd.read_csv(matrix_path, index_col=0)
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    # Фрагмент матрицы для самых населённых кварталов — как acc_mx.head() в pipeline.ipynb.
    sample_ids = [int(i) for i in blocks.nlargest(8, "population").index if i in matrix.index]
    matrix_info = {
        "size": [int(values.shape[0]), int(values.shape[1])],
        "median": clean(np.median(finite), 1) if finite.size else None,
        "max": clean(finite.max(), 1) if finite.size else None,
        "unreachableShare": clean(float(np.isinf(values).mean()), 4),
        "ids": sample_ids,
        "values": [[clean(matrix.at[a, b], 1) for b in sample_ids] for a in sample_ids],
    }

    # Связи «квартал → школа» с временем в пути.
    links = links.copy()
    if len(links):
        links["time_min"] = [matrix.at[s, t] for s, t in zip(links["source"], links["target"])]
    links = links.sort_values("value", ascending=False)

    clusters = clusters.copy()
    sizes = blocks["cluster"].dropna().astype(int).value_counts()
    clusters["blocks"] = clusters["cluster"].map(sizes).fillna(0).astype(int)

    developed = blocks[(blocks["morphotype_strelka"] != "no_data")]
    crosstab = pd.crosstab(developed["morphotype_strelka"], developed["morphotype_spacematrix"])

    raw_dates = {}
    for name in ["boundary", "roads", "railways", "water", "functional_zones", "buildings_raw", "schools_raw"]:
        path = RAW_DIR / f"{name}.geojson"
        if path.exists():
            raw_dates[name] = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d.%m.%Y %H:%M")

    bounds = blocks_wgs.total_bounds
    return {
        "territory": TERRITORY_LABEL,
        "generated": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "summary": {k: clean(v, 4) for k, v in summary.items()},
        "summaryRows": records(summary_df, 4),
        "metrics": metrics,
        "blocks": geojson,
        "boundary": json.loads(boundary.to_crs(4326).to_json()),
        "schools": school_features(school_points),
        "bounds": [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        "landArea": [{"value": k, "label": LAND_USE_LABELS.get(k, k), "ha": clean(v, 1)} for k, v in land_area.items()],
        "provisionTotals": {
            "demand": clean(blocks["demand"].sum()),
            "capacity": clean(blocks["capacity"].sum()),
            "within": clean(blocks["demand_within"].sum()),
            "without": clean(blocks["demand_without"].sum()),
            "left": clean(blocks["demand_left"].sum()),
            "capacityLeft": clean(blocks["capacity_left"].sum()),
        },
        "links": records(links, 1),
        "matrix": matrix_info,
        "clusters": records(clusters, 3),
        "crosstab": {
            "rows": [str(i) for i in crosstab.index],
            "columns": [str(c) for c in crosstab.columns],
            "values": crosstab.values.astype(int).tolist(),
        },
        "rawDates": raw_dates,
        "conclusions": build_conclusions(blocks, summary),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Выборг: кварталы и метрики</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --neutral: #e6e5df;
  --accent: #2a78d6;
  --critical: #d03b3b;
  --good: #006300;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7;
    --text-muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --neutral: #383835; --accent: #3987e5; --critical: #e66767; --good: #0ca30c;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7;
  --text-muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --neutral: #383835; --accent: #3987e5; --critical: #e66767; --good: #0ca30c;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header, main { max-width: 1280px; margin: 0 auto; padding: 0 16px; }
header { padding-top: 28px; padding-bottom: 8px; }
h1 { font-size: 28px; margin: 0 0 4px; }
h2 { font-size: 20px; margin: 36px 0 12px; }
h3 { font-size: 15px; margin: 0 0 8px; }
p { margin: 0 0 8px; }
.muted { color: var(--text-secondary); }
.small { font-size: 13px; }
nav { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 12px 0 0; font-size: 14px; }
nav a { color: var(--accent); text-decoration: none; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 16px; }
.tile .value { font-size: 26px; font-weight: 600; }
.tile .label { font-size: 13px; color: var(--text-secondary); }
.badge { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 4px;
  border: 1px solid var(--border); color: var(--text-secondary); margin-left: 4px; vertical-align: middle; }
.map-layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 12px; }
#map { height: 640px; border-radius: 10px; border: 1px solid var(--border); }
.controls { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin-bottom: 10px; }
select, input[type=search] { font: inherit; padding: 6px 8px; border-radius: 6px; border: 1px solid var(--axis);
  background: var(--surface-1); color: var(--text-primary); }
#metric-select { min-width: 280px; max-width: 100%; }
.legend-row { display: flex; align-items: center; gap: 8px; font-size: 13px; margin: 3px 0; }
.swatch { width: 14px; height: 14px; border-radius: 3px; flex: none; border: 1px solid var(--border); }
#details dl { display: grid; grid-template-columns: 1fr auto; gap: 2px 10px; margin: 0; font-size: 13px; }
#details dt { color: var(--text-secondary); }
#details dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
#details .group { grid-column: 1 / -1; font-weight: 600; margin-top: 8px; color: var(--text-primary); }
.side { display: flex; flex-direction: column; gap: 12px; max-height: 640px; overflow: auto; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 12px; }
.chart-box { position: relative; height: 300px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 5px 8px; border-bottom: 1px solid var(--grid); text-align: left; vertical-align: top; }
th { color: var(--text-secondary); font-weight: 600; position: sticky; top: 0; background: var(--surface-1); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.table-wrap { overflow: auto; max-height: 520px; }
#blocks-table th { cursor: pointer; white-space: nowrap; }
#blocks-table tbody tr { cursor: pointer; }
#blocks-table tbody tr:hover { background: var(--neutral); }
.pager { display: flex; gap: 8px; align-items: center; margin-top: 8px; font-size: 13px; }
.pager button { font: inherit; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--axis);
  background: var(--surface-1); color: var(--text-primary); cursor: pointer; }
.conclusions { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
.conclusion .kind { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.conclusion.finding .kind { color: var(--accent); }
.conclusion.caveat .kind { color: var(--critical); }
footer { max-width: 1280px; margin: 40px auto 24px; padding: 0 16px; font-size: 13px; color: var(--text-secondary); }
.leaflet-tooltip { font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
@media (max-width: 900px) {
  .map-layout { grid-template-columns: 1fr; }
  #map { height: 460px; }
  .side { max-height: none; }
  .charts { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>Выборг: кварталы и метрики BlocksNet</h1>
  <p class="muted">Учебное исследование по открытым данным OpenStreetMap. Страница собрана <span id="generated"></span>
    из готовых результатов ноутбука <code>vyborg_blocksnet_study.ipynb</code>.</p>
  <nav>
    <a href="#overview">Обзор</a><a href="#map-section">Карта</a><a href="#charts">Графики</a>
    <a href="#conclusions">Выводы</a><a href="#tables">Таблицы</a><a href="#dictionary">Показатели</a>
    <a href="#sources">Источники и ограничения</a>
  </nav>
</header>
<main>
  <section id="overview"><div class="tiles" id="tiles"></div></section>

  <section id="map-section">
    <h2>Карта кварталов</h2>
    <div class="controls">
      <label>Показатель: <select id="metric-select"></select></label>
      <label><input type="checkbox" id="show-schools" checked> школы</label>
    </div>
    <div class="map-layout">
      <div id="map"></div>
      <div class="side">
        <div class="card"><h3 id="legend-title"></h3><p class="small muted" id="legend-desc"></p><div id="legend"></div></div>
        <div class="card" id="details"><h3>Квартал</h3><p class="small muted">Щёлкните по кварталу на карте или по строке таблицы.</p></div>
      </div>
    </div>
  </section>

  <section id="charts">
    <h2>Графики</h2>
    <div class="charts">
      <div class="card"><h3>Площадь по функциям, га</h3><div class="chart-box"><canvas id="chart-land"></canvas></div></div>
      <div class="card"><h3>Школьные места: потребность и наличие</h3><div class="chart-box"><canvas id="chart-provision"></canvas></div></div>
      <div class="card"><h3>Обеспеченность strong в жилых кварталах</h3><p class="small muted">Число кварталов с населением по долям обеспеченности.</p><div class="chart-box"><canvas id="chart-provision-hist"></canvas></div></div>
      <div class="card"><h3>Население: 15 самых населённых кварталов</h3><div class="chart-box"><canvas id="chart-population"></canvas></div></div>
      <div class="card"><h3>Морфотипы Strelka, кварталов</h3><div class="chart-box"><canvas id="chart-strelka"></canvas></div></div>
      <div class="card"><h3>Морфотипы Spacematrix, кварталов</h3><div class="chart-box"><canvas id="chart-spacematrix"></canvas></div></div>
      <div class="card"><h3>Диаграмма Spacematrix: GSI и FSI</h3><p class="small muted">Каждая точка — застроенный квартал.</p><div class="chart-box"><canvas id="chart-scatter"></canvas></div></div>
      <div class="card"><h3>Медианное время пешком до остальных кварталов</h3><p class="small muted">Число кварталов по интервалам 10 минут.</p><div class="chart-box"><canvas id="chart-time"></canvas></div></div>
    </div>
  </section>

  <section id="conclusions">
    <h2>Выводы и ограничения</h2>
    <div class="conclusions" id="conclusion-list"></div>
  </section>

  <section id="tables">
    <h2>Таблицы</h2>
    <div class="card">
      <h3>Все показатели по кварталам</h3>
      <div class="controls">
        <label>Группа: <select id="table-group"></select></label>
        <label>Поиск: <input type="search" id="table-search" placeholder="номер квартала или тип"></label>
      </div>
      <div class="table-wrap"><table id="blocks-table"><thead></thead><tbody></tbody></table></div>
      <div class="pager"><button id="prev">←</button><span id="page-info"></span><button id="next">→</button></div>
    </div>
    <div class="charts" style="margin-top:12px">
      <div class="card"><h3>Матрица доступности, минуты пешком</h3><p class="small muted" id="matrix-info"></p><div class="table-wrap"><table id="matrix-table"></table></div></div>
      <div class="card"><h3>Школы</h3><p class="small muted">Вместимость по OSM или подставленная BlocksNet (250 мест).</p><div class="table-wrap"><table id="schools-table"></table></div></div>
      <div class="card"><h3>Сводка расчёта</h3><div class="table-wrap"><table id="summary-table"></table></div></div>
      <div class="card"><h3>Потоки детей между кварталами</h3><p class="small muted">Сколько мест (value) жители квартала source получили в школах квартала target.</p><div class="table-wrap"><table id="links-table"></table></div></div>
      <div class="card"><h3>Группы Spacematrix</h3><p class="small muted">Медианы l, fsi, mxi в группе и число кварталов.</p><div class="table-wrap"><table id="clusters-table"></table></div></div>
      <div class="card"><h3>Strelka × Spacematrix</h3><p class="small muted">Сколько кварталов каждого типа Strelka попало в каждую группу Spacematrix.</p><div class="table-wrap"><table id="crosstab-table"></table></div></div>
    </div>
  </section>

  <section id="dictionary">
    <h2>Что означают показатели</h2>
    <div class="card table-wrap" style="max-height:none"><table id="dictionary-table"></table></div>
  </section>

  <section id="sources">
    <h2>Источники и ограничения</h2>
    <div class="card" id="sources-box"></div>
  </section>
</main>
<footer>Данные © участники OpenStreetMap, лицензия ODbL. Расчёты — BlocksNet 1.0.0a9. Подложка карты — CARTO.</footer>

<script id="dashboard-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("dashboard-data").textContent);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const DIV = [
  { to: 0.2, color: "#b8312f", label: "0–0,2 — сильный дефицит" },
  { to: 0.4, color: "#e34948", label: "0,2–0,4" },
  { to: 0.6, color: "#d9d8d3", label: "0,4–0,6" },
  { to: 0.8, color: "#86b6ef", label: "0,6–0,8" },
  { to: 1.0001, color: "#2a78d6", label: "0,8–1 — мест хватает" },
];
const ZERO_NEUTRAL = new Set(["count", "is_living", "floors_from_osm", "footprint_area", "build_floor_area",
  "living_area", "non_living_area", "population", "schools_count", "capacity", "capacity_osm", "capacity_estimated",
  "capacity_imputed", "demand", "demand_within", "demand_without", "demand_left", "capacity_within",
  "capacity_without", "capacity_left", "residential", "business", "recreation", "industrial", "transport",
  "special", "agriculture", "fsi", "gsi"]);
const metricsByKey = Object.fromEntries(DATA.metrics.map((m) => [m.key, m]));
const features = DATA.blocks.features;
const nf = (value, digits) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (typeof value !== "number") return String(value);
  const d = digits !== undefined ? digits : (Math.abs(value) >= 100 ? 0 : Math.abs(value) >= 10 ? 1 : 2);
  return value.toLocaleString("ru-RU", { maximumFractionDigits: d, minimumFractionDigits: 0 });
};
const slotColor = (slot) => (slot ? css(`--s${slot}`) : css("--neutral"));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const displayValue = (metric, value) => {
  if (metric && metric.kind === "cat" && metric.categories) {
    const cat = metric.categories.find((c) => c.value === value);
    return cat ? cat.label : value;
  }
  return nf(value);
};

document.getElementById("generated").textContent = DATA.generated;

// ---------- плитки
(function tiles() {
  const s = DATA.summary;
  const p = DATA.provisionTotals;
  const items = [
    [nf(s.population_total, 0), "жителей", "оценка"],
    [nf(s.blocks_count, 0), "кварталов", ""],
    [nf(s.buildings_count, 0), "зданий в OSM", ""],
    [nf(s.schools_count, 0), "школ", ""],
    [`${nf(p.capacity, 0)} / ${nf(p.demand, 0)}`, "мест в школах / нужно по нормативу", "оценка"],
    [`${nf(p.demand ? (p.within / p.demand) * 100 : 0, 0)} %`, "потребности закрыто в пределах 15 мин", "модель"],
    [`${nf(s.median_travel_time_min, 0)} мин`, "медианное время пешком между кварталами", ""],
  ];
  document.getElementById("tiles").innerHTML = items.map(([v, l, b]) =>
    `<div class="card tile"><div class="value">${esc(v)}</div><div class="label">${esc(l)}${b ? `<span class="badge">${b}</span>` : ""}</div></div>`
  ).join("");
})();

// ---------- карта
const map = L.map("map", { preferCanvas: true });
const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
L.tileLayer(`https://{s}.basemaps.cartocdn.com/${dark ? "dark_all" : "light_all"}/{z}/{x}/{y}{r}.png`, {
  attribution: "&copy; участники OpenStreetMap &copy; CARTO", subdomains: "abcd", maxZoom: 19,
}).addTo(map);
map.fitBounds(DATA.bounds);
L.geoJSON(DATA.boundary, { style: { color: css("--text-primary"), weight: 1.5, fill: false }, interactive: false }).addTo(map);

let currentMetric = "provision_strong";
let colorFor = () => css("--neutral");
let selectedLayer = null;

function quantileBreaks(values, classes) {
  const sorted = values.slice().sort((a, b) => a - b);
  const breaks = [];
  for (let i = 1; i < classes; i++) {
    const v = sorted[Math.min(sorted.length - 1, Math.floor((i / classes) * sorted.length))];
    if (!breaks.length || v > breaks[breaks.length - 1]) breaks.push(v);
  }
  return breaks;
}

function buildScale(metric) {
  const legend = [];
  if (metric.kind === "cat") {
    const colors = Object.fromEntries(metric.categories.map((c) => [c.value, slotColor(c.slot)]));
    metric.categories.forEach((c) => legend.push({ color: slotColor(c.slot), label: c.label }));
    return { fn: (v) => colors[v] || css("--neutral"), legend };
  }
  if (metric.kind === "div") {
    DIV.forEach((d) => legend.push({ color: d.color, label: d.label }));
    legend.push({ color: css("--neutral"), label: "нет жителей", empty: true });
    return {
      fn: (v) => (v === null || v === undefined ? "zero" : DIV.find((d) => v < d.to).color),
      legend,
    };
  }
  const zeroNeutral = ZERO_NEUTRAL.has(metric.key);
  const values = features.map((f) => f.properties[metric.key])
    .filter((v) => typeof v === "number" && (!zeroNeutral || v > 0));
  if (!values.length) {
    legend.push({ color: css("--neutral"), label: "все значения 0 или нет данных" });
    return { fn: () => null, legend };
  }
  const breaks = quantileBreaks(values, SEQ.length);
  const classes = breaks.length + 1;
  const ramp = Array.from({ length: classes }, (_, i) =>
    SEQ[classes === 1 ? SEQ.length - 1 : Math.round((i * (SEQ.length - 1)) / (classes - 1))]);
  const min = Math.min(...values), max = Math.max(...values);
  const edges = [min, ...breaks, max];
  for (let i = 0; i < ramp.length; i++) {
    legend.push({ color: ramp[i], label: `${nf(edges[i])} – ${nf(edges[i + 1])}` });
  }
  if (zeroNeutral) legend.push({ color: css("--neutral"), label: "0", empty: true });
  legend.push({ color: "transparent", label: "нет данных", empty: true, hollow: true });
  return {
    fn: (v) => {
      if (v === null || v === undefined) return null;
      if (zeroNeutral && v <= 0) return "zero";
      let i = 0;
      while (i < breaks.length && v >= breaks[i]) i++;
      return ramp[Math.min(i, ramp.length - 1)];
    },
    legend,
  };
}

function styleFor(feature) {
  const c = colorFor(feature.properties[currentMetric]);
  if (c === null) return { color: css("--axis"), weight: 0.4, fillOpacity: 0, fillColor: "transparent" };
  if (c === "zero") return { color: css("--surface-1"), weight: 0.3, fillColor: css("--neutral"), fillOpacity: 0.55 };
  return { color: css("--surface-1"), weight: 0.4, fillColor: c, fillOpacity: 0.85 };
}

const blocksLayer = L.geoJSON(DATA.blocks, {
  style: styleFor,
  onEachFeature: (feature, layer) => {
    layer.bindTooltip(() => {
      const m = metricsByKey[currentMetric];
      const p = feature.properties;
      return `<b>Квартал ${p.block_id}</b><br>${esc(m.label)}: <b>${esc(displayValue(m, p[currentMetric]))}</b>` +
        (m.unit && m.kind !== "cat" ? ` ${esc(m.unit)}` : "") +
        `<br><span style="opacity:.7">население ${nf(p.population, 0)} · ${esc(displayValue(metricsByKey.land_use, p.land_use))}</span>`;
    }, { sticky: true });
    layer.on("mouseover", () => layer.setStyle({ weight: 2, color: css("--text-primary") }));
    layer.on("mouseout", () => { if (layer !== selectedLayer) blocksLayer.resetStyle(layer); });
    layer.on("click", () => selectBlock(feature.properties.block_id, false));
  },
}).addTo(map);
const layerById = {};
blocksLayer.eachLayer((layer) => { layerById[layer.feature.properties.block_id] = layer; });

const schoolsLayer = L.layerGroup(DATA.schools.map((s) => L.circleMarker([s.lat, s.lon], {
  radius: 6, weight: 2, color: css("--surface-1"), fillColor: s.imputed ? css("--s2") : css("--text-primary"), fillOpacity: 1,
}).bindTooltip(`Школа: ${nf(s.capacity, 0)} мест${s.imputed ? " (вместимость подставлена)" : " (из OSM)"}`))).addTo(map);
document.getElementById("show-schools").addEventListener("change", (e) => {
  if (e.target.checked) schoolsLayer.addTo(map); else map.removeLayer(schoolsLayer);
});

function renderLegend(metric, scale) {
  document.getElementById("legend-title").textContent = `${metric.label}${metric.kind !== "cat" ? `, ${metric.unit}` : ""}`;
  document.getElementById("legend-desc").textContent = `${metric.description} Источник: ${metric.source}.`;
  const rows = scale.legend.map((r) =>
    `<div class="legend-row"><span class="swatch" style="background:${r.color};${r.hollow ? `border-color:${css("--axis")}` : ""}"></span>${esc(r.label)}</div>`);
  const schoolRows = `<div class="legend-row"><span class="swatch" style="border-radius:50%;background:${css("--text-primary")}"></span>школа, вместимость из OSM</div>` +
    `<div class="legend-row"><span class="swatch" style="border-radius:50%;background:${css("--s2")}"></span>школа, вместимость подставлена</div>`;
  document.getElementById("legend").innerHTML = rows.join("") + schoolRows;
}

function setMetric(key) {
  currentMetric = key;
  const metric = metricsByKey[key];
  const scale = buildScale(metric);
  colorFor = scale.fn;
  blocksLayer.setStyle(styleFor);
  if (selectedLayer) selectedLayer.setStyle({ weight: 3, color: css("--text-primary") });
  renderLegend(metric, scale);
}

(function metricSelect() {
  const select = document.getElementById("metric-select");
  const groups = {};
  DATA.metrics.filter((m) => m.kind !== "tech").forEach((m) => { (groups[m.group] = groups[m.group] || []).push(m); });
  select.innerHTML = Object.entries(groups).map(([g, ms]) =>
    `<optgroup label="${esc(g)}">${ms.map((m) => `<option value="${m.key}">${esc(m.label)}</option>`).join("")}</optgroup>`).join("");
  select.value = currentMetric;
  select.addEventListener("change", () => setMetric(select.value));
})();

function selectBlock(blockId, zoom) {
  const layer = layerById[blockId];
  if (!layer) return;
  if (selectedLayer) blocksLayer.resetStyle(selectedLayer);
  selectedLayer = layer;
  layer.setStyle({ weight: 3, color: css("--text-primary") });
  layer.bringToFront();
  if (zoom) {
    map.fitBounds(layer.getBounds(), { maxZoom: 16 });
    document.getElementById("map-section").scrollIntoView({ behavior: "smooth" });
  }
  const p = layer.feature.properties;
  let html = `<h3>Квартал ${p.block_id}</h3><dl>`;
  let group = null;
  DATA.metrics.forEach((m) => {
    if (m.group !== group) { group = m.group; html += `<div class="group">${esc(group)}</div>`; }
    html += `<dt>${esc(m.label)}${m.kind !== "cat" && m.unit !== "—" ? `, ${esc(m.unit)}` : ""}</dt><dd>${esc(displayValue(m, p[m.key]))}</dd>`;
  });
  document.getElementById("details").innerHTML = html + "</dl>";
}

setMetric(currentMetric);

// ---------- графики
Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif';
Chart.defaults.color = css("--text-secondary");
Chart.defaults.borderColor = css("--grid");
Chart.defaults.plugins.legend.display = false;
const barBase = { borderRadius: 4, borderSkipped: "start", maxBarThickness: 28, categoryPercentage: 0.8, barPercentage: 0.9 };
const axes = (horizontal, titleX, titleY) => ({
  x: { grid: { display: !horizontal ? false : true }, title: { display: !!titleX, text: titleX }, beginAtZero: true,
       ticks: { callback: function (v) { return horizontal ? nf(v) : this.getLabelForValue(v); } } },
  y: { grid: { display: horizontal ? false : true }, title: { display: !!titleY, text: titleY }, beginAtZero: true,
       ticks: { callback: function (v) { return horizontal ? this.getLabelForValue(v) : nf(v); } } },
});
function barChart(id, labels, values, colors, opts = {}) {
  const horizontal = !!opts.horizontal;
  return new Chart(document.getElementById(id), {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors, ...barBase }] },
    options: {
      indexAxis: horizontal ? "y" : "x", responsive: true, maintainAspectRatio: false,
      scales: axes(horizontal, opts.titleX, opts.titleY),
      plugins: { tooltip: { callbacks: { label: (ctx) => ` ${nf(ctx.parsed[horizontal ? "x" : "y"])}${opts.unit ? " " + opts.unit : ""}` } } },
    },
  });
}
function histogram(values, step, count) {
  const bins = Array.from({ length: count }, (_, i) => ({ from: i * step, to: (i + 1) * step, n: 0 }));
  values.forEach((v) => { const i = Math.min(count - 1, Math.floor(v / step + 1e-9)); if (i >= 0) bins[i].n++; });
  return bins;
}

const landCats = metricsByKey.land_use.categories;
barChart("chart-land", DATA.landArea.map((d) => d.label), DATA.landArea.map((d) => d.ha),
  DATA.landArea.map((d) => slotColor((landCats.find((c) => c.value === d.value) || {}).slot)), { horizontal: true, unit: "га", titleX: "га" });

const pt = DATA.provisionTotals;
barChart("chart-provision", ["Нужно мест", "Мест в школах", "Закрыто ≤ 15 мин", "Не хватает"],
  [pt.demand, pt.capacity, pt.within, pt.left], [css("--s1"), css("--s1"), css("--s1"), css("--critical")], { unit: "мест", titleY: "места" });

const populated = features.map((f) => f.properties).filter((p) => p.population > 0 && p.provision_strong !== null);
const provBins = histogram(populated.map((p) => p.provision_strong), 0.1, 10);
barChart("chart-provision-hist", provBins.map((b) => `${nf(b.from, 1)}–${nf(b.to, 1)}`), provBins.map((b) => b.n),
  provBins.map((b) => DIV.find((d) => b.from + 0.05 < d.to).color), { unit: "кварталов", titleY: "кварталов", titleX: "provision_strong" });

const topPop = features.map((f) => f.properties).filter((p) => p.population > 0).sort((a, b) => b.population - a.population).slice(0, 15);
barChart("chart-population", topPop.map((p) => `квартал ${p.block_id}`), topPop.map((p) => p.population), css("--s1"), { horizontal: true, unit: "чел.", titleX: "человек" });

function morphChart(id, key) {
  const m = metricsByKey[key];
  const counts = {};
  features.forEach((f) => { const v = f.properties[key]; if (v && v !== "no_data") counts[v] = (counts[v] || 0) + 1; });
  const cats = m.categories.filter((c) => c.value !== "no_data");
  barChart(id, cats.map((c) => c.label), cats.map((c) => counts[c.value] || 0), cats.map((c) => slotColor(c.slot)), { horizontal: true, unit: "кварталов" });
}
morphChart("chart-strelka", "morphotype_strelka");
morphChart("chart-spacematrix", "morphotype_spacematrix");

const developed = features.map((f) => f.properties).filter((p) => p.fsi > 0 && p.gsi !== null);
new Chart(document.getElementById("chart-scatter"), {
  type: "scatter",
  data: { datasets: [{ data: developed.map((p) => ({ x: p.gsi, y: p.fsi, id: p.block_id, t: p.morphotype_spacematrix })),
    backgroundColor: css("--s1"), borderColor: css("--surface-1"), borderWidth: 1, pointRadius: 4, pointHoverRadius: 7 }] },
  options: {
    responsive: true, maintainAspectRatio: false,
    scales: { x: { title: { display: true, text: "GSI — доля застроенной земли" }, beginAtZero: true },
              y: { title: { display: true, text: "FSI — площадь этажей / площадь квартала" }, beginAtZero: true } },
    plugins: { tooltip: { callbacks: { label: (ctx) => ` квартал ${ctx.raw.id}: GSI ${nf(ctx.raw.x)}, FSI ${nf(ctx.raw.y)} · ${ctx.raw.t}` } } },
    onClick: (_, elements) => { if (elements.length) selectBlock(developed[elements[0].index].block_id, true); },
  },
});

const times = features.map((f) => f.properties.median_time_min).filter((v) => typeof v === "number");
const timeBins = histogram(times, 10, Math.max(1, Math.ceil(Math.max(...times) / 10)));
barChart("chart-time", timeBins.map((b) => `${b.from}–${b.to}`), timeBins.map((b) => b.n), css("--s1"), { unit: "кварталов", titleX: "минуты", titleY: "кварталов" });

// ---------- выводы
document.getElementById("conclusion-list").innerHTML = DATA.conclusions.map((c) =>
  `<div class="card conclusion ${c.kind}"><div class="kind">${c.kind === "finding" ? "Вывод" : "Ограничение"}</div><h3>${esc(c.title)}</h3><p class="small">${esc(c.text)}</p></div>`).join("");

// ---------- таблицы
function simpleTable(id, columns, rows) {
  const head = `<thead><tr>${columns.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("")}</tr></thead>`;
  const body = rows.map((r) => `<tr>${columns.map((c) => `<td class="${c.num ? "num" : ""}">${esc(c.num ? nf(r[c.key]) : (r[c.key] ?? "—"))}</td>`).join("")}</tr>`).join("");
  document.getElementById(id).innerHTML = head + `<tbody>${body}</tbody>`;
}
(function matrixTable() {
  const m = DATA.matrix;
  document.getElementById("matrix-info").textContent =
    `Размер ${m.size[0]} × ${m.size[1]} кварталов. Медиана ${nf(m.median, 1)} мин, максимум ${nf(m.max, 0)} мин, ` +
    `пар без маршрута ${nf(m.unreachableShare * 100, 2)} %. Ниже — фрагмент для 8 самых населённых кварталов: строка — откуда, столбец — куда.`;
  const head = `<thead><tr><th>из \\ в</th>${m.ids.map((id) => `<th class="num">${id}</th>`).join("")}</tr></thead>`;
  const body = m.ids.map((id, i) => `<tr><td>${id}</td>${m.values[i].map((v) => `<td class="num">${nf(v, 1)}</td>`).join("")}</tr>`).join("");
  document.getElementById("matrix-table").innerHTML = head + `<tbody>${body}</tbody>`;
})();
simpleTable("schools-table", [{ key: "block_id", label: "квартал" }, { key: "capacity", label: "мест", num: true },
  { key: "source", label: "вместимость" }], DATA.schools.map((s) => ({ ...s, source: s.imputed ? "подставлена" : "из OSM" })));
simpleTable("summary-table", [{ key: "metric", label: "показатель" }, { key: "value", label: "значение" }], DATA.summaryRows);
simpleTable("links-table", [{ key: "source", label: "source" }, { key: "target", label: "target" }, { key: "value", label: "мест", num: true }, { key: "time_min", label: "минут пешком", num: true }], DATA.links);
simpleTable("clusters-table", [{ key: "cluster", label: "группа" }, { key: "morphotype", label: "морфотип" }, { key: "l", label: "l", num: true }, { key: "fsi", label: "fsi", num: true }, { key: "mxi", label: "mxi", num: true }, { key: "blocks", label: "кварталов", num: true }], DATA.clusters);
(function crosstab() {
  const ct = DATA.crosstab;
  const head = `<thead><tr><th>Strelka \\ Spacematrix</th>${ct.columns.map((c) => `<th class="num">${esc(c)}</th>`).join("")}</tr></thead>`;
  const body = ct.rows.map((r, i) => `<tr><td>${esc(r)}</td>${ct.values[i].map((v) => `<td class="num">${v || ""}</td>`).join("")}</tr>`).join("");
  document.getElementById("crosstab-table").innerHTML = head + `<tbody>${body}</tbody>`;
})();
simpleTable("dictionary-table", [{ key: "group", label: "группа" }, { key: "key", label: "колонка" }, { key: "label", label: "название" },
  { key: "unit", label: "единицы" }, { key: "source", label: "откуда" }, { key: "description", label: "что означает" }], DATA.metrics);

(function blocksTable() {
  const groupSelect = document.getElementById("table-group");
  const groups = [...new Set(DATA.metrics.map((m) => m.group))];
  groupSelect.innerHTML = groups.map((g) => `<option>${esc(g)}</option>`).join("");
  const rowsAll = features.map((f) => f.properties);
  let sortKey = "block_id", sortDir = 1, page = 0;
  const pageSize = 25;
  const render = () => {
    const cols = [{ key: "block_id", label: "квартал", kind: "seq" }, ...DATA.metrics.filter((m) => m.group === groupSelect.value)];
    const q = document.getElementById("table-search").value.trim().toLowerCase();
    let rows = rowsAll.filter((r) => !q || String(r.block_id) === q || cols.some((c) => typeof r[c.key] === "string" && r[c.key].toLowerCase().includes(q)));
    rows.sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      if (x === y) return 0;
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return (x > y ? 1 : -1) * sortDir;
    });
    const pages = Math.max(1, Math.ceil(rows.length / pageSize));
    page = Math.min(page, pages - 1);
    const shown = rows.slice(page * pageSize, page * pageSize + pageSize);
    const table = document.getElementById("blocks-table");
    table.tHead.innerHTML = `<tr>${cols.map((c) => `<th class="${c.kind === "cat" ? "" : "num"}" data-key="${c.key}">${esc(c.label)}${c.unit && c.kind !== "cat" ? `, ${esc(c.unit)}` : ""}${sortKey === c.key ? (sortDir > 0 ? " ↑" : " ↓") : ""}</th>`).join("")}</tr>`;
    table.tBodies[0].innerHTML = shown.map((r) => `<tr data-id="${r.block_id}">${cols.map((c) =>
      `<td class="${c.kind === "cat" ? "" : "num"}">${esc(c.key === "block_id" ? r.block_id : displayValue(c, r[c.key]))}</td>`).join("")}</tr>`).join("");
    document.getElementById("page-info").textContent = `стр. ${page + 1} из ${pages} · кварталов: ${rows.length}`;
  };
  document.getElementById("blocks-table").tHead.addEventListener("click", (e) => {
    const key = e.target.closest("th")?.dataset.key;
    if (!key) return;
    sortDir = key === sortKey ? -sortDir : (key === "block_id" ? 1 : -1);
    sortKey = key; render();
  });
  document.getElementById("blocks-table").tBodies[0].addEventListener("click", (e) => {
    const id = e.target.closest("tr")?.dataset.id;
    if (id) selectBlock(Number(id), true);
  });
  groupSelect.addEventListener("change", () => { page = 0; render(); });
  document.getElementById("table-search").addEventListener("input", () => { page = 0; render(); });
  document.getElementById("prev").addEventListener("click", () => { page = Math.max(0, page - 1); render(); });
  document.getElementById("next").addEventListener("click", () => { page += 1; render(); });
  render();
})();

// ---------- источники
(function sources() {
  const s = DATA.summary;
  const dates = Object.entries(DATA.rawDates).map(([k, v]) => `<li><code>${esc(k)}</code> — загружено ${esc(v)}</li>`).join("");
  document.getElementById("sources-box").innerHTML = `
    <p><b>Данные:</b> OpenStreetMap через Overpass API, граница — OSM relation ${nf(s.territory_osm_id, 0)} (город ${esc(DATA.territory)}).</p>
    <ul class="small">${dates}</ul>
    <p><b>Граф:</b> пешеходная сеть OpenStreetMap, скорость ${nf(s.walk_speed_kmh, 1)} км/ч, время в минутах. Интермодальный граф не строился.</p>
    <p><b>Расчёты:</b> BlocksNet ${esc(s.blocksnet_version)}, как в examples/pipeline.ipynb. Нормативы школ: 120 мест на 1000 жителей, 15 минут пешком.</p>
    <p><b>Оценки, а не исходные данные:</b> население, площадь этажей, жилая и нежилая площадь, вместимость школ без тега capacity
      (250 мест, на этой странице — оранжевые школы и колонка «Подставленные места»), этажность зданий без тега building:levels (1 этаж).</p>
    <p><b>Ограничения:</b> OSM неполный; граф не учитывает заборы и закрытые территории; функция квартала выбирается по наибольшей
      доле даже при 1–2 %; морфотипы зависят от этажности, которой в OSM почти нет.</p>`;
})();
</script>
</body>
</html>
"""


def main():
    payload = build_payload()
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA__", data_json)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    size_mb = OUTPUT_PATH.stat().st_size / 1_000_000
    print(f"Готово: {OUTPUT_PATH}")
    print(f"Размер файла: {size_mb:.1f} МБ. Откройте его двойным щелчком в браузере.")


if __name__ == "__main__":
    main()
