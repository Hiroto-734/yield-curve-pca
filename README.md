# Yield Curve PCA Analysis

> 米国債イールドカーブを主成分分析(PCA)で構造分解し、金利市場の動きを `Level / Slope / Curvature` の3軸で解釈する学習プロジェクト

## Status

🚧 **In progress** — Phase 1 (Notebook 学習) を作業中

詳細な仕様は [`yield_curve_pca_spec.md`](./yield_curve_pca_spec.md) を参照。

## Quick start

```bash
# 仮想環境の作成と依存インストール
python -m uv venv
source .venv/Scripts/activate    # Windows (Git Bash)
# source .venv/bin/activate      # macOS / Linux

python -m uv pip install pandas numpy scikit-learn requests \
    matplotlib seaborn jupyterlab pyarrow pytest ruff mypy

# Notebook 起動
jupyter lab
```

## Project structure

```
.
├── data/
│   ├── raw/           # FRED から取得した生データ(.gitignore 対象)
│   └── processed/     # 前処理済み parquet
├── notebooks/         # Phase 1: 学習用 Notebook
├── src/               # Phase 2: モジュール化(後で実装)
├── tests/             # Phase 2: テスト
├── reports/
│   ├── figures/       # 出力グラフ
│   └── findings.md    # 発見の言語化
└── docs/
    ├── glossary.md    # 金融用語集
    └── learning_log.md # 学びのログ
```
