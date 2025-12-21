from __future__ import annotations

import pandas as pd

from src.eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    ColumnSummary,
    DatasetSummary,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_new_quality_flags(): # Специальный DataFrame для проверки всех новых эвристик
    n = 60  # число строк
    df = pd.DataFrame({
        "user_id": list(range(1, n)) + [2],  # дубликат -> has_suspicious_id_duplicates=True
        "numeric_col": [0] * n,  # все нули -> has_many_zero_values=True
        "constant_col": [5] * n,  # константа -> has_constant_columns=True
        "category_col": [f"A{i}" for i in range(n)],  # >50 уникальных -> has_high_cardinality_categoricals=True
    })

    columns = []
    for col in df.columns:
        s = df[col]
        columns.append(
            ColumnSummary(
                name=col,
                dtype=str(s.dtype),
                non_null=s.notna().sum(),
                missing=len(s) - s.notna().sum(),
                missing_share=0.0,
                unique=s.nunique(),
                example_values=s.dropna().astype(str).unique()[:3].tolist(),
                is_numeric=pd.api.types.is_numeric_dtype(s),
                min=s.min() if pd.api.types.is_numeric_dtype(s) else None,
                max=s.max() if pd.api.types.is_numeric_dtype(s) else None,
                mean=s.mean() if pd.api.types.is_numeric_dtype(s) else None,
                std=s.std() if pd.api.types.is_numeric_dtype(s) else None,
            )
        )

    summary = DatasetSummary(n_rows=len(df), n_cols=len(df.columns), columns=columns)
    missing_df = pd.DataFrame({"missing_share": [0.0]*len(df.columns)})
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_constant_columns"] is True
    assert flags["has_high_cardinality_categoricals"] is True
    assert flags["has_suspicious_id_duplicates"] is True
    assert flags["has_many_zero_values"] is True
