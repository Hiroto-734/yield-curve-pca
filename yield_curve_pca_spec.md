# Yield Curve PCA Analysis Project

> 米国債イールドカーブを主成分分析(PCA)で構造分解し、金利市場の動きを `Level / Slope / Curvature` の3軸で解釈できるようにする学習プロジェクト

---

## 1. プロジェクトの目的

### 1.1 ゴール
**「金利市場の日々の動きを、3つの解釈可能な数字に圧縮する分析パイプライン」を自分の手で作り、結果を金融的に語れるようになる。**

### 1.2 サブゴール(段階的に達成)
1. イールドカーブとは何かを、データとグラフで体得する
2. PCA という数理手法を金融データに適用できるようになる
3. 結果を金融用語(Level / Slope / Curvature, ベア・スティープ等)で語れるようになる
4. 就活で見せられるポートフォリオ(GitHub + 日本語READMEの解説)に仕上げる

### 1.3 非ゴール(やらないこと)
- バックテスト戦略の構築(将来のフェーズで実施)
- リアルタイムデータ取得・自動化(本プロジェクトでは日次データで十分)
- 多通貨・他国債への展開(まずは米国債のみに集中)

---

## 2. 学んで得られるスキル

| カテゴリ | 内容 |
|---|---|
| 金融知識 | イールドカーブ、期間構造、bp、ベア/ブル × フラット/スティープ |
| データ分析 | 時系列差分処理、欠損処理、標準化 |
| 機械学習 | PCAの理論と実装、寄与率、固有ベクトル解釈 |
| 可視化 | 静的グラフ(matplotlib)、必要に応じインタラクティブ(plotly) |
| エンジニアリング | プロジェクト構成、再現可能性、Notebook → モジュール化 |
| 金融的解釈 | マクロイベントとPCAスコアの照合、市場状態の言語化 |

---

## 3. プロジェクト構成

### 3.1 2フェーズ構成

```
Phase 1: 学習(Notebook主体)─── まず理解する
  └─ notebooks/01_explore.ipynb など
                ↓
Phase 2: 整理(モジュール化)─── 見せられる形にする
  └─ src/ にロジック移行、tests/ で検証
```

**重要**: Phase 1 を完成させてから Phase 2 に進む。両方を同時にやらない。

### 3.2 ディレクトリ構造

```
yield-curve-pca/
├── README.md                          # プロジェクト概要(英語+日本語)
├── pyproject.toml                     # 依存管理(poetry or uv 推奨)
├── .gitignore
├── .python-version                    # 3.11 を推奨
│
├── data/
│   ├── raw/                           # FREDからダウンロードした生CSV
│   │   └── ust_yields_2020_2026.csv
│   └── processed/                     # 前処理後のparquet
│       └── ust_yields_clean.parquet
│
├── notebooks/                         # Phase 1: 学習用
│   ├── 01_data_exploration.ipynb      # データ取得と可視化
│   ├── 02_curve_dynamics.ipynb        # カーブの動きを観察
│   ├── 03_pca_basics.ipynb            # PCAの適用と解釈
│   ├── 04_pca_interpretation.ipynb    # Level/Slope/Curvatureの理解
│   └── 05_event_study.ipynb           # FOMC等のイベント分析
│
├── src/                               # Phase 2: モジュール化
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # FREDデータ取得
│   │   └── preprocessor.py            # 差分・bp変換等
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── pca_analyzer.py            # PCA本体
│   │   └── interpreter.py             # 結果の金融的解釈
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── curve_plots.py             # カーブ描画
│   │   └── pca_plots.py               # 主成分の可視化
│   └── utils/
│       ├── __init__.py
│       └── config.py                  # 定数・設定
│
├── tests/                             # Phase 2: テスト
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   └── test_pca_analyzer.py
│
├── reports/                           # 成果物
│   ├── figures/                       # 主要グラフ(PNG)
│   └── findings.md                    # 発見事項のサマリー(日本語OK)
│
└── docs/
    ├── glossary.md                    # 金融用語集
    └── learning_log.md                # 自分の学びを書き残す
```

### 3.3 重要な設計判断

| 判断 | 理由 |
|---|---|
| データを `parquet` で保存 | CSVより軽量・高速、型情報を保持 |
| Notebook と src を分離 | 学習用と再利用可能ロジックを混ぜない |
| `reports/findings.md` を別出し | 就活で見せるサマリー資料 |
| `learning_log.md` を残す | 後から振り返って言語化できる |

---

## 4. 環境構築

### 4.1 推奨環境
- Python 3.11
- パッケージ管理: `uv`(高速)または `poetry`
- エディタ: VSCode + Jupyter拡張、または JupyterLab

### 4.2 依存パッケージ

```toml
[project]
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "pandas-datareader>=0.10",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "jupyterlab>=4.0",
    "pyarrow>=14.0",          # parquet用
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "ruff>=0.1",              # linter & formatter
    "mypy>=1.5",
]
```

### 4.3 セットアップコマンド

```bash
# uvを使う場合(推奨)
uv venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# pip派なら
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 5. Phase 1: Notebook学習フロー

### Notebook 01: データ探索 `01_data_exploration.ipynb`

**目的**: データを取得し、構造を把握する

**実装内容**:
1. FREDから米国債10年限のデータを取得(2020年~現在)
2. 欠損値の確認と処理
3. 各年限の時系列プロット
4. `data/raw/` と `data/processed/` に保存

**チェックポイント**:
- [ ] データの shape は `(N営業日, 10年限)` になっているか
- [ ] 欠損日数を把握したか(連休・米国祝日)
- [ ] 各年限の時系列が「金利」として妥当な範囲(0%〜6%程度)か

**観察してほしいこと**:
- 2022年からのFRB急激な利上げで全カーブが上方シフトしている
- 2y と 10y の差が一時的にマイナス(逆イールド)になった時期がある

---

### Notebook 02: カーブダイナミクス `02_curve_dynamics.ipynb`

**目的**: イールドカーブの「形の変化」を目で理解する

**実装内容**:
1. 単一日のカーブ描画(横軸=年限、縦軸=利回り)
2. 月次スナップショットを重ね描き
3. 日次変化(差分)を bp 単位で可視化
4. 「Slope = 10y - 2y」を時系列でプロット

**チェックポイント**:
- [ ] カーブが「右肩上がりが普通だが、形が日々違う」ことを目視確認
- [ ] 1日の動きが大体 ±10bp の範囲に収まることを確認
- [ ] 2022〜2024年あたりで Slope がマイナス(逆イールド)になったことを確認

**金融用語の導入**:
- 順イールド / 逆イールド
- ベア・スティープ / ブル・フラットの4分類
- スプレッドトレード(2s10s フラットナー等)

---

### Notebook 03: PCAの基礎 `03_pca_basics.ipynb`

**目的**: PCA を実装し、3軸が現れることを確認する

**実装内容**:
1. 日次変化を計算し、bp単位に変換
2. `sklearn.decomposition.PCA(n_components=3)` で分解
3. 寄与率を確認(Level=85%, Slope=10%, Curvature=3%が目安)
4. 各成分のローディング(年限ごとの重み)を可視化

**チェックポイント**:
- [ ] PC1の寄与率が80%以上
- [ ] PC1のローディングが全年限で同符号(=Level)
- [ ] PC2のローディングが短期と長期で逆符号(=Slope)
- [ ] PC3のローディングが両端と中間で逆符号(=Curvature)

**学びの確認**:
- なぜPCAは差分(変化)に対してかけるのか?
- 標準化(StandardScaler)はかけるべきか? → かけない方が金融的解釈がしやすい(議論あり)

---

### Notebook 04: PCAの解釈 `04_pca_interpretation.ipynb`

**目的**: 各PCスコアを時系列で見て、金融的に解釈する

**実装内容**:
1. 全期間のPC1/PC2/PC3 スコアを時系列プロット
2. PCスコアと「人間が決めた指標」の比較
   - PC1 vs 10y金利
   - PC2 vs (10y - 2y) スプレッド
   - PC3 vs 「バタフライ = 2×5y - (2y+10y)」
3. 相関係数の計算

**チェックポイント**:
- [ ] PC1 と 10y金利の相関が0.95以上
- [ ] PC2 とスロープの相関が0.9以上
- [ ] PC3 とバタフライの相関が0.8以上

**学びの確認**:
- PCA は「人間が決めた指標」とほぼ同じものをデータから見つけ出している
- PCA の利点は「客観性」と「最適性」(分散最大化の意味で)

---

### Notebook 05: イベントスタディ `05_event_study.ipynb`

**目的**: 実際の市場イベントを PCA で説明する

**実装内容**:
1. FOMC日付リスト(2022年以降)を作成
2. 各FOMC日のPC1/PC2/PC3スコアを抽出
3. 「サプライズあり/なし」で分類(オプション)
4. CPI発表日、雇用統計日でも同じ分析

**ケーススタディ例**:
- 2022年6月のFOMC(75bp利上げ): PC1の大きなプラス
- 2024年3月SVB破綻: PC1の急落+PC2のスティープ化
- 2025年9月の利下げ転換期: PC2の大きな動き

**成果物**:
- `reports/findings.md` に「3つの代表的イベント」の解説を書く

---

## 6. Phase 2: モジュール化

### 6.1 移行のタイミング
- Notebook 01-05 がすべて動いている
- 結果に納得できている
- 同じコードを2回以上コピペした感覚がある

### 6.2 モジュール設計の原則

**Single Responsibility**: 各モジュールは1つの責任を持つ

```python
# src/data/loader.py
def fetch_treasury_yields(
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """FREDから米国債利回りを取得する"""
    ...

# src/data/preprocessor.py  
def to_daily_changes_bp(yields: pd.DataFrame) -> pd.DataFrame:
    """日次変化を bp 単位に変換する"""
    ...

# src/analysis/pca_analyzer.py
class YieldCurvePCA:
    def fit(self, changes_bp: pd.DataFrame) -> "YieldCurvePCA":
        ...
    def transform(self, changes_bp: pd.DataFrame) -> pd.DataFrame:
        ...
    @property
    def loadings(self) -> pd.DataFrame:
        ...
```

### 6.3 テスト方針

最低限以下をテスト:
- データ取得関数: モックデータで動作確認
- 差分変換: 既知の入出力で確認
- PCA: 寄与率の合計が1になる、shapeが正しい

```python
# tests/test_pca_analyzer.py
def test_pca_explains_most_variance():
    pca = YieldCurvePCA(n_components=3)
    pca.fit(sample_data)
    assert pca.explained_variance_ratio_.sum() > 0.95
```

---

## 7. 成果物の整理(就活で見せる形)

### 7.1 README.md(プロジェクトTOP)

以下の構成にする:

1. **What it does**(1段落)
2. **Demo図**(主要なグラフ1〜2枚を埋め込む)
3. **Key findings**(箇条書き3〜5項目)
4. **How to run**(再現手順)
5. **Project structure**(ディレクトリ説明)
6. **What I learned**(技術と金融の学び)

### 7.2 reports/findings.md

「自分の言葉で解釈した結果」を日本語で書く。面接で口頭説明する**台本**になる。

```markdown
# 主な発見

## 1. PC1 = Level の解釈
- 寄与率: 87.3%
- 10年金利との相関: 0.97
- 大きく動いた日トップ3: ...
- 解釈: FRBの政策金利期待を反映...

## 2. PC2 = Slope の解釈
...
```

### 7.3 GitHubリポジトリの整え方

- LICENSEを置く(MITで十分)
- `.gitignore` に `data/raw/`, `*.ipynb_checkpoints/` を含める
- ブランチを切って開発(`main` は常に動く状態に保つ)
- Commit message を意味のある単位にまとめる

---

## 8. 学習ログのテンプレート

`docs/learning_log.md` に毎回書き残す:

```markdown
## 2026-05-XX
### やったこと
- Notebook 01 完了

### 詰まったこと
- DGS3MO に欠損が多い → 補間 vs 削除で迷った → 削除で進めた

### 新しく知った金融用語
- 順イールド: 通常の状態。長期金利 > 短期金利
- 逆イールド: 短期 > 長期。景気後退の予兆とされる

### 次にやること
- Notebook 02 のカーブ重ね描き
```

これがあると面接で「**学習プロセスを言語化できる人**」という印象になる。

---

## 9. タイムライン目安

| 期間 | やること |
|---|---|
| Day 1 | 環境構築、Notebook 01 |
| Day 2 | Notebook 02 |
| Day 3 | Notebook 03(PCA本体) |
| Day 4 | Notebook 04(解釈) |
| Day 5 | Notebook 05(イベントスタディ) |
| Day 6 | Phase 2: モジュール化 |
| Day 7 | README整備、findings.md執筆、GitHub公開 |

**合計約1週間**(1日2〜3時間想定)。研究と就活の合間に集中して取り組めば現実的。

---

## 10. 発展課題(余裕があれば)

将来のフェーズとして:
- [ ] **平均回帰戦略**: PC2(Slope)の Z-score がしきい値超でフラットナー/スティープナー
- [ ] **クロスカントリー比較**: 米国 vs ドイツ国債のPCA比較
- [ ] **インプライドボラ**: スワップションのインプライドボラとPCAスコアの関係
- [ ] **マクロファクター回帰**: PC1〜3 を CPI、雇用統計サプライズで回帰

---

## 11. 参考リソース

### 必読
- Litterman & Scheinkman (1991) "Common Factors Affecting Bond Returns" — PCAでの金利分析の元祖論文
- Fabozzi『フィクスト・インカム証券』第2〜4章

### Web
- FRED: https://fred.stlouisfed.org/
- US Treasury Daily Yield Curve Rates(公式)

### 金融用語のチェック
- 日本証券アナリスト協会の用語集
- Bloomberg Terminal用語(無料記事レベルで十分)

---

## 12. このプロジェクトを面接で語るときの型

> 「FXのアルゴ取引経験から**マイクロ構造には強かった**のですが、FICC全体を志望するにあたり**金利市場の構造的理解**を深めたいと考え、米国債イールドカーブをPCA分解するプロジェクトをやりました。
> 
> 過去5年の日次データに対してPCAをかけたところ、**寄与率上位3軸が全体の98%を説明**し、それぞれ Level・Slope・Curvature と解釈できました。これは Litterman & Scheinkman (1991) の古典的結果と整合的です。
> 
> 例えば 2024年3月のSVB破綻時には PC1 が急落し PC2 がスティープ化しており、これは『短期金利低下期待 + 長期インフレ警戒』というマーケットの解釈と一致していました。
> 
> 今後はこの3軸の**平均回帰性を使った戦略**まで拡張していきたいと考えています。」

---

**最終チェックリスト(プロジェクト完了の定義)**:

- [ ] Notebook 01-05 がすべてエラーなく実行できる
- [ ] `reports/figures/` に主要グラフ5枚以上ある
- [ ] `reports/findings.md` に3つの発見が書かれている
- [ ] README.md が英日両言語で書かれている
- [ ] GitHubに公開されている
- [ ] 面接で5分間口頭説明できる
