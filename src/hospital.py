"""
Загрузка и предобработка данных о госпитализациях.

Используется в ноутбуке notebooks/hospital_analysis.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "hospital.csv"

COLUMNS = {
    "ДАТА_ПОСТУПЛЕНИЯ": "admission_date",
    "ДАТА_ВЫПИСКИ": "discharge_date",
    "УНИКАЛЬНЫЙ_ID": "patient_id",
    "НОМЕР_ИБ": "case_number",
    "ШИФР": "payer_code",
    "РЕГИОН": "region",
    "ПЛАТЕЛЬЩИК": "payer",
    "ИСТОЧНИК_ФИНАНСИРОВАНИЯ": "funding",
    "ОТДЕЛЕНИЕ": "department",
    "ПРОФИЛЬ": "profile",
    "КОЙКО_ДНИ": "bed_days",
    "ПЛАНОВО_ЭКСТРЕННО": "admission_type",
    "КОД_МКБ_ПРИ_ПОСТУПЛЕНИИ": "icd_admission",
    "КОД_МКБ_ПРИ_ВЫПИСКЕ": "icd_discharge",
}

CATEGORY_COLUMNS = ["region", "payer", "funding", "department", "profile", "admission_type"]

# Отделение, где пациентов наблюдают не дольше суток и затем переводят или отпускают
OBSERVATION_DEPARTMENT = "ПЭМП ДПО"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Читает исходный файл без преобразований, чтобы сначала изучить его как есть."""
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {path.name}. Положите файл в data/raw/ (см. data/raw/README.md)."
        )
    return pd.read_csv(path)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Короткие латинские имена вместо длинных кириллических; проверяет, что все столбцы известны."""
    unknown = set(df.columns) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Неизвестные столбцы: {sorted(unknown)}")
    return df.rename(columns=COLUMNS)


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Даты -> datetime, койко-дни -> int, текстовые признаки с малым числом значений -> category,
    идентификаторы и коды -> строки (с ними не нужно считать).
    """
    out = df.copy()
    for col in ["admission_date", "discharge_date"]:
        out[col] = pd.to_datetime(out[col], format="%d.%m.%Y")
    out["bed_days"] = out["bed_days"].astype(int)
    for col in ["patient_id", "case_number", "payer_code"]:
        out[col] = out[col].astype(str)
    for col in CATEGORY_COLUMNS:
        out[col] = out[col].astype("category")
    return out


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Регион не указан более чем у половины пациентов — явно отмечаем это, а не удаляем строки.
    Строки без кода МКБ (их единицы) оставляем: для длительности госпитализации диагноз не нужен.
    """
    out = df.copy()
    out["region"] = out["region"].cat.add_categories("Не указан").fillna("Не указан")
    for col in ["icd_admission", "icd_discharge"]:
        out[col] = out[col].fillna("не указан")
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки для анализа: год и месяц поступления, флаги дневного стационара и отделения наблюдения."""
    out = df.copy()
    out["year"] = out["admission_date"].dt.year
    out["month"] = out["admission_date"].dt.month
    out["is_day_hospital"] = out["department"].astype(str).str.contains("Дневной стационар")
    out["is_observation"] = out["department"].astype(str).eq(OBSERVATION_DEPARTMENT)
    return out


def prepare(path: Path = RAW_PATH) -> pd.DataFrame:
    """Полный конвейер: загрузка -> переименование -> дубликаты -> типы -> пропуски -> признаки."""
    df = rename_columns(load_raw(path)).drop_duplicates().reset_index(drop=True)
    return add_features(fill_missing(convert_types(df)))
