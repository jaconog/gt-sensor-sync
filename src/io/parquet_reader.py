# src/io/parquet_reader.py
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
import pandas as pd

BASE_PARQUET_PATH = os.getenv("BASE_PARQUET_PATH", "data/raw")


@lru_cache(maxsize=10)
def read_trace_parquet(trace_id: str, base_path: str = BASE_PARQUET_PATH) -> Optional[pd.DataFrame]:
    """Lee el archivo Parquet de un trace_id específico usando la estructura de GT-Mapper.
    Mantiene los últimos 10 archivos en caché de RAM.
    """
    parquet_file = os.path.join(
        base_path, str(trace_id), f"trace_master_{trace_id}.parquet"
    )
    if os.path.exists(parquet_file):
        return pd.read_parquet(parquet_file)
    return None


@lru_cache(maxsize=10)
def read_parquet_by_path(file_path: str) -> pd.DataFrame:
    """Lee cualquier archivo Parquet pasando su ruta absoluta o relativa directamente.
    Útil cuando se trabaja con archivos fuera del esquema trace_id/trace_master.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo Parquet en: {file_path}")

    return pd.read_parquet(path)