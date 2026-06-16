"""
=============================================================
  ГЛАВА 3. ПОСТРОЕНИЕ И АНАЛИЗ ПРЕДСКАЗАТЕЛЬНЫХ МОДЕЛЕЙ
  Разделы 3.1–3.4
  Автор: Сиянко М.М.
=============================================================

КАК ЗАПУСКАТЬ:
  1. Укажи пути к файлам в блоке НАСТРОЙКА ПУТЕЙ (см. ниже).
  2. Запусти: python chapter3_analysis.py
  3. Все рисунки сохранятся рядом со скриптом:
       fig1_actual_vs_pred_A.png   — Рис. 1 (Модель А, факт vs прогноз)
       fig2_feature_importance_A.png — Рис. 2 (топ-10 признаков А)
       fig3_actual_vs_pred_B.png   — Рис. 3 (Модель Б, факт vs прогноз)
       fig4_feature_importance_B.png — Рис. 4 (топ-10 признаков Б)
       fig5_residuals_A.png        — Рис. 5 (анализ остатков А)
       fig6_forecast_2025.png      — Рис. 6 (прогноз 2025, но данные 25-го года
                                             отсутствуют → раздел закомментирован)
       fig7_decision_tree.png      — Рис. 7 (дерево решений)
       fig8_confusion_matrix.png   — Рис. 8 (матрица ошибок лучшей модели)
       fig9_feature_importance_clf.png — Рис. 9 (важность признаков RF-CLF)
  4. В консоль выводятся все числовые результаты (Таблицы 2–6).
=============================================================
"""

# ─── НАСТРОЙКА ПУТЕЙ ──────────────────────────────────────
DATASET_REGRESSION = "dataset без 25 года.xlsx"   # ← поправь при необходимости
DATASET_DEVIANT    = "dataset_deviant_50_final.xlsx"
OUTPUT_DIR         = "."                            # куда сохранять рисунки
# ──────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (r2_score, mean_absolute_error,
                             mean_squared_error,
                             accuracy_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                     LeaveOneOut, GridSearchCV)

# ─── Стиль графиков ───────────────────────────────────────
rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":     150,
})
PALETTE = "#2E75B6"

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ═══════════════════════════════════════════════════════════
# ЧАСТЬ 1.  РЕГРЕССИОННЫЕ МОДЕЛИ (разделы 3.1–3.2)
# ═══════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  ЧАСТЬ 1. РЕГРЕССИОННЫЕ МОДЕЛИ")
print("═"*60)

# ─── Загрузка данных ──────────────────────────────────────
df_reg = pd.read_excel(DATASET_REGRESSION)
df_reg.rename(columns={"Unnamed: 0": "Регион"}, inplace=True)

# Очищаем названия столбцов
df_reg.columns = df_reg.columns.str.strip().str.replace(r"\s+", " ", regex=True)

TARGET_REG = "Количество несовершеннолетних, совершивших преступления"
YEAR_COL   = "Год"

# Проверка наличия целевой переменной
assert TARGET_REG in df_reg.columns, \
    f"Не найден столбец целевой переменной. Проверь: {df_reg.columns.tolist()}"

# ─── Формирование признаков ───────────────────────────────
# Признаки, общие для обеих моделей (А и Б)
FEATS_INFRA = [
    "Число домашних хозяйств  всего, тыс. Единиц",   # масштаб
    "компьютер (настольный, мобильный, планшетный)",  # оборудование
    "доступ в Интернет",                               # инфраструктура
    "доступ в Интернет с компьютера",
    "широкополосный доступ к сети Интернет",
    "Настольные (стационарные) компьютеры",
    "Мобильные компьютеры (лэптопы, ноутбуки, нетбуки, ультрабуки)",
    "Планшетные компьютеры",
    "Мобильный телефон или смартфон",
    "Телевизоры со специальным устройством (Smart TV)",
    "Отсутствие доступа к сети Интернет",
    "15  19 лет",                                      # демография
]

FEATS_BEHAV = [
    "прочтение новостной информации, статей",
    "общение в социальных сетях для поддержания личных контактов",
    "обсуждение социальных и политических вопросов, участие в "
    "интернетакциях, опросах общественного мнения и др.",
    "скачивание фильмов, музыки и игр, сетевые игры и др.",
]

# Очищаем пробелы в именах после strip (некоторые столбцы начинаются с пробела)
col_map = {}
for c in df_reg.columns:
    col_map[c.strip()] = c
    col_map[c.lstrip()] = c

def find_col(keyword, cols):
    """Ищет столбец по ключевому слову (частичное совпадение)."""
    for c in cols:
        if keyword.lower() in c.lower():
            return c
    return None

# Автоматически находим нужные столбцы по ключевым словам
kw_infra = [
    "хозяйств",
    "компьютер (настольный",
    "доступ в Интернет",
    "доступ в Интернет с компьютера",
    "широкополосный",
    "Настольные (стационарные)",
    "Мобильные компьютеры",
    "Планшетные компьютеры",
    "Мобильный телефон",
    "Smart TV",
    "Отсутствие доступа",
    "15",
]
kw_behav = [
    "прочтение новостной",
    "общение в социальных сетях",
    "обсуждение социальных",
    "скачивание фильмов",
]

feats_infra_found = []
for kw in kw_infra:
    c = find_col(kw, df_reg.columns)
    if c and c not in feats_infra_found:
        feats_infra_found.append(c)

feats_behav_found = []
for kw in kw_behav:
    c = find_col(kw, df_reg.columns)
    if c and c not in feats_behav_found:
        feats_behav_found.append(c)

print(f"\nИнфраструктурных признаков найдено: {len(feats_infra_found)}")
print(f"Поведенческих признаков найдено: {len(feats_behav_found)}")

# ─── Короткие метки для графиков ──────────────────────────
LABELS_A = {
    feats_infra_found[0]:  "Домашних хозяйств, тыс.",
    feats_infra_found[1]:  "Наличие компьютера, %",
    feats_infra_found[2]:  "Доступ в Интернет, %",
    feats_infra_found[3]:  "Интернет с компьютера, %",
    feats_infra_found[4]:  "Широкополосный доступ, %",
    feats_infra_found[5]:  "Настольные ПК, %",
    feats_infra_found[6]:  "Мобильные ПК, %",
    feats_infra_found[7]:  "Планшеты, %",
    feats_infra_found[8]:  "Смартфон/телефон, %",
    feats_infra_found[9]:  "Smart TV, %",
    feats_infra_found[10]: "Нет доступа в Интернет, %",
    feats_infra_found[11]: "Численность 15–19 лет",
}

LABELS_B_extra = {}
label_behav_names = [
    "Чтение новостей, %",
    "Общение в соцсетях, %",
    "Обсуждение общ. вопросов, %",
    "Развлекательный контент, %",
]
for col, lbl in zip(feats_behav_found, label_behav_names):
    LABELS_B_extra[col] = lbl

LABELS_ALL = {**LABELS_A, **LABELS_B_extra}

def short_label(col):
    return LABELS_ALL.get(col, col[:30])


# ─── Разделение: Модель А (регионы), Модель Б (округа) ───
df_regions = df_reg[df_reg["Тип территории"] == "регион"].copy()
df_feds    = df_reg[df_reg["Тип территории"] == "федеральный округ"].copy()

# Добавляем год как числовой признак (линейное кодирование)
for df_ in [df_regions, df_feds]:
    df_["Год_num"] = (df_[YEAR_COL] - df_[YEAR_COL].min())

YEAR_FEAT = "Год_num"

# ─── Разбивка train/val ───────────────────────────────────
TRAIN_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
VAL_YEARS   = [2023, 2024]

def split_tv(df_, feat_cols, target):
    tr = df_[df_[YEAR_COL].isin(TRAIN_YEARS)]
    va = df_[df_[YEAR_COL].isin(VAL_YEARS)]
    Xtr, ytr = tr[feat_cols].values, tr[target].values
    Xva, yva = va[feat_cols].values, va[target].values
    return Xtr, ytr, Xva, yva


# ─── Функция обучения трёх моделей ───────────────────────
def train_regressors(Xtr, ytr, Xva, yva, feat_names, model_tag=""):
    results = {}
    importances = {}

    # --- Ridge ---
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)

    param_grid = {"alpha": [0.01, 0.1, 1, 10, 100, 500]}
    ridge = GridSearchCV(Ridge(), param_grid, cv=5, scoring="r2")
    ridge.fit(Xtr_s, ytr)
    best_ridge = ridge.best_estimator_
    yp = best_ridge.predict(Xva_s)
    results["Ridge"] = {
        "R2":   round(r2_score(yva, yp), 4),
        "MAE":  round(mean_absolute_error(yva, yp), 2),
        "RMSE": round(np.sqrt(mean_squared_error(yva, yp)), 2),
        "pred": yp,
        "coef": best_ridge.coef_,
        "alpha": ridge.best_params_["alpha"],
    }
    importances["Ridge"] = dict(zip(feat_names, np.abs(best_ridge.coef_)))

    # --- Lasso ---
    lasso = GridSearchCV(Lasso(max_iter=10000), param_grid, cv=5, scoring="r2")
    lasso.fit(Xtr_s, ytr)
    best_lasso = lasso.best_estimator_
    yp = best_lasso.predict(Xva_s)
    results["Lasso"] = {
        "R2":   round(r2_score(yva, yp), 4),
        "MAE":  round(mean_absolute_error(yva, yp), 2),
        "RMSE": round(np.sqrt(mean_squared_error(yva, yp)), 2),
        "pred": yp,
        "alpha": lasso.best_params_["alpha"],
    }
    importances["Lasso"] = dict(zip(feat_names, np.abs(best_lasso.coef_)))

    # --- Random Forest ---
    rf_param = {
        "n_estimators": [100, 200],
        "max_depth":    [3, 5, 7, None],
        "min_samples_split": [2, 5],
    }
    rf = GridSearchCV(RandomForestRegressor(random_state=42),
                      rf_param, cv=5, scoring="r2", n_jobs=-1)
    rf.fit(Xtr, ytr)
    best_rf = rf.best_estimator_
    yp = best_rf.predict(Xva)
    results["Random Forest"] = {
        "R2":   round(r2_score(yva, yp), 4),
        "MAE":  round(mean_absolute_error(yva, yp), 2),
        "RMSE": round(np.sqrt(mean_squared_error(yva, yp)), 2),
        "pred": yp,
        "params": rf.best_params_,
    }
    importances["Random Forest"] = dict(zip(feat_names, best_rf.feature_importances_))

    return results, importances, best_rf, scaler


# ══════════════════════════════════════════
#  МОДЕЛЬ А — регионы (96 субъектов РФ)
# ══════════════════════════════════════════
print("\n" + "─"*40)
print("МОДЕЛЬ А — регионы (96 субъектов РФ)")
print("─"*40)

feats_A = feats_infra_found + [YEAR_FEAT]

# Удаляем строки с NaN в нужных столбцах
df_A = df_regions[feats_A + [TARGET_REG, YEAR_COL]].dropna()
feat_names_A = [short_label(f) if f != YEAR_FEAT else "Год" for f in feats_A]

Xtr_A, ytr_A, Xva_A, yva_A = split_tv(df_A, feats_A, TARGET_REG)

print(f"  Train: {len(Xtr_A)} obs | Val: {len(Xva_A)} obs")

res_A, imp_A, best_rf_A, scaler_A = train_regressors(
    Xtr_A, ytr_A, Xva_A, yva_A, feat_names_A, "A"
)

print("\n  Метрики Модели А (тест 2023–2024):")
for algo, m in res_A.items():
    print(f"    {algo:15s}  R²={m['R2']:6.4f}  MAE={m['MAE']:8.1f}  RMSE={m['RMSE']:8.1f}")

# ─── Рис. 1 — Факт vs Прогноз (RF, Модель А) ─────────────
yp_rf_A = res_A["Random Forest"]["pred"]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(yva_A, yp_rf_A, alpha=0.6, color=PALETTE, edgecolors="white", s=50)
mn, mx = min(yva_A.min(), yp_rf_A.min()), max(yva_A.max(), yp_rf_A.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Идеальный прогноз")
ax.set_xlabel("Фактическое значение (чел.)")
ax.set_ylabel("Прогноз (чел.)")
ax.set_title("Рис. 1 — Факт vs. прогноз: Модель А (регионы, Random Forest)")
ax.legend()
ax.text(0.05, 0.92, f"R² = {res_A['Random Forest']['R2']:.4f}",
        transform=ax.transAxes, fontsize=10, color="darkred")
savefig("fig1_actual_vs_pred_A.png")

# ─── Рис. 2 — Топ-10 признаков Модели А ───────────────────
imp_rf_A = imp_A["Random Forest"]
top10_A  = sorted(imp_rf_A.items(), key=lambda x: x[1], reverse=True)[:10]
lbls_A, vals_A = zip(*top10_A)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(range(len(lbls_A)), vals_A, color=PALETTE, alpha=0.85)
ax.set_yticks(range(len(lbls_A)))
ax.set_yticklabels(lbls_A, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Gini importance")
ax.set_title("Рис. 2 — Топ-10 признаков по важности: Модель А (Random Forest)")
for bar, val in zip(bars, vals_A):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
savefig("fig2_feature_importance_A.png")


# ══════════════════════════════════════════
#  МОДЕЛЬ Б — федеральные округа
# ══════════════════════════════════════════
print("\n" + "─"*40)
print("МОДЕЛЬ Б — федеральные округа (8 ФО)")
print("─"*40)

feats_B = feats_infra_found + feats_behav_found + [YEAR_FEAT]

df_B = df_feds[feats_B + [TARGET_REG, YEAR_COL]].dropna()
feat_names_B = [short_label(f) if f != YEAR_FEAT else "Год" for f in feats_B]

Xtr_B, ytr_B, Xva_B, yva_B = split_tv(df_B, feats_B, TARGET_REG)

print(f"  Train: {len(Xtr_B)} obs | Val: {len(Xva_B)} obs")

res_B, imp_B, best_rf_B, scaler_B = train_regressors(
    Xtr_B, ytr_B, Xva_B, yva_B, feat_names_B, "B"
)

print("\n  Метрики Модели Б (тест 2023–2024):")
for algo, m in res_B.items():
    print(f"    {algo:15s}  R²={m['R2']:6.4f}  MAE={m['MAE']:8.1f}  RMSE={m['RMSE']:8.1f}")

# ─── Рис. 3 — Факт vs Прогноз (RF, Модель Б) ─────────────
yp_rf_B = res_B["Random Forest"]["pred"]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(yva_B, yp_rf_B, alpha=0.7, color="#ED7D31", edgecolors="white", s=70)
mn, mx = min(yva_B.min(), yp_rf_B.min()), max(yva_B.max(), yp_rf_B.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Идеальный прогноз")
ax.set_xlabel("Фактическое значение (чел.)")
ax.set_ylabel("Прогноз (чел.)")
ax.set_title("Рис. 3 — Факт vs. прогноз: Модель Б (федеральные округа, Random Forest)")
ax.legend()
ax.text(0.05, 0.92, f"R² = {res_B['Random Forest']['R2']:.4f}",
        transform=ax.transAxes, fontsize=10, color="darkred")
savefig("fig3_actual_vs_pred_B.png")

# ─── Рис. 4 — Топ-10 признаков Модели Б ───────────────────
imp_rf_B = imp_B["Random Forest"]
top10_B  = sorted(imp_rf_B.items(), key=lambda x: x[1], reverse=True)[:10]
lbls_B, vals_B = zip(*top10_B)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(range(len(lbls_B)), vals_B, color="#ED7D31", alpha=0.85)
ax.set_yticks(range(len(lbls_B)))
ax.set_yticklabels(lbls_B, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Gini importance")
ax.set_title("Рис. 4 — Топ-10 признаков по важности: Модель Б (Random Forest)")
for bar, val in zip(bars, vals_B):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
savefig("fig4_feature_importance_B.png")


# ─── Рис. 5 — Анализ остатков (лучшая модель А) ───────────
# Определяем лучшую модель А по R²
best_algo_A = max(res_A, key=lambda k: res_A[k]["R2"])
yp_best_A   = res_A[best_algo_A]["pred"]
residuals_A = yva_A - yp_best_A

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Остатки vs прогноз
axes[0].scatter(yp_best_A, residuals_A, alpha=0.6, color=PALETTE, edgecolors="white", s=50)
axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
axes[0].set_xlabel("Прогнозируемое значение (чел.)")
axes[0].set_ylabel("Остатки (факт − прогноз)")
axes[0].set_title(f"Остатки vs. прогноз — Модель А ({best_algo_A})")

# Гистограмма остатков
axes[1].hist(residuals_A, bins=20, color=PALETTE, alpha=0.7, edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
axes[1].set_xlabel("Остатки")
axes[1].set_ylabel("Частота")
axes[1].set_title(f"Распределение остатков — Модель А ({best_algo_A})")

plt.suptitle("Рис. 5 — Анализ остатков: Модель А", fontsize=13, y=1.01)
savefig("fig5_residuals_A.png")


# ─── Сводная Таблица 2 ────────────────────────────────────
print("\n" + "═"*60)
print("  ТАБЛИЦА 2 — Сводные метрики регрессионных моделей (тест 2023–2024)")
print("═"*60)
print(f"{'Алгоритм':<18} {'Данные':<22} {'R²':>8} {'MAE':>10} {'RMSE':>10}")
print("-"*72)
for algo in ["Ridge", "Lasso", "Random Forest"]:
    m = res_A[algo]
    print(f"  {algo:<16} {'Регионы (А)':<22} {m['R2']:>8.4f} {m['MAE']:>10.1f} {m['RMSE']:>10.1f}")
for algo in ["Ridge", "Lasso", "Random Forest"]:
    m = res_B[algo]
    print(f"  {algo:<16} {'Фед. округа (Б)':<22} {m['R2']:>8.4f} {m['MAE']:>10.1f} {m['RMSE']:>10.1f}")
print()


# ─── Таблица 3 — Топ-5 признаков ──────────────────────────
best_algo_B = max(res_B, key=lambda k: res_B[k]["R2"])

imp_top5_A = sorted(imp_A[best_algo_A].items(), key=lambda x: x[1], reverse=True)[:5]
imp_top5_B = sorted(imp_B[best_algo_B].items(), key=lambda x: x[1], reverse=True)[:5]

print("═"*60)
print("  ТАБЛИЦА 3 — Топ-5 значимых признаков")
print("═"*60)
print(f"  {'Ранг':>4}  {'Признак (А)':<30} {'Важн. А':>8}  {'Признак (Б)':<30} {'Важн. Б':>8}")
print("  " + "-"*88)
for i, ((fa, va), (fb, vb)) in enumerate(zip(imp_top5_A, imp_top5_B), 1):
    print(f"  {i:>4}  {fa:<30} {va:>8.4f}  {fb:<30} {vb:>8.4f}")
print()


# ═══════════════════════════════════════════════════════════
# ЧАСТЬ 2.  КЛАССИФИКАЦИОННАЯ МОДЕЛЬ (разделы 3.3–3.4)
# ═══════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  ЧАСТЬ 2. КЛАССИФИКАЦИОННАЯ МОДЕЛЬ (девиантный риск)")
print("═"*60)

# ─── Загрузка датасета ────────────────────────────────────
df_dev = pd.read_excel(DATASET_DEVIANT, header=2)

TARGET_CLF = "Уровень девиантного риска"
DROP_COLS   = ["ID"]

# Кодируем категориальные
le_gender = LabelEncoder()
df_dev["Пол"] = le_gender.fit_transform(df_dev["Пол"])          # М=1, Ж=0

le_income = LabelEncoder()
df_dev["Доход семьи"] = le_income.fit_transform(df_dev["Доход семьи"])

le_family = LabelEncoder()
df_dev["Полная семья"] = le_family.fit_transform(df_dev["Полная семья"])

# Целевая переменная
le_target = LabelEncoder()
df_dev[TARGET_CLF] = le_target.fit_transform(df_dev[TARGET_CLF])
class_names = le_target.classes_   # ['Высокий', 'Низкий', 'Средний']

feat_cols_clf = [c for c in df_dev.columns if c not in DROP_COLS + [TARGET_CLF]]

X_clf = df_dev[feat_cols_clf].values.astype(float)
y_clf = df_dev[TARGET_CLF].values

print(f"\n  Признаков: {X_clf.shape[1]} | Наблюдений: {X_clf.shape[0]}")
print(f"  Классы: {class_names}")

# ─── 5-fold StratifiedKFold CV ───────────────────────────
skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
loocv = LeaveOneOut()

scaler_clf = MinMaxScaler()
X_clf_scaled = scaler_clf.fit_transform(X_clf)

def clf_cv_metrics(model, X, y, cv):
    acc  = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
    mf1  = cross_val_score(model, X, y, cv=cv, scoring="f1_macro").mean()
    wf1  = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted").mean()
    return round(acc, 4), round(mf1, 4), round(wf1, 4)

# ─── Decision Tree ────────────────────────────────────────
dt = DecisionTreeClassifier(max_depth=4, criterion="gini",
                             class_weight="balanced", random_state=42)
acc_dt_5,  mf1_dt_5,  wf1_dt_5  = clf_cv_metrics(dt, X_clf,        y_clf, skf)
acc_dt_lo, mf1_dt_lo, wf1_dt_lo = clf_cv_metrics(dt, X_clf,        y_clf, loocv)

# ─── Logistic Regression ──────────────────────────────────
lr = LogisticRegression(C=1.0, multi_class="ovr", class_weight="balanced",
                         max_iter=5000, random_state=42)
acc_lr_5,  mf1_lr_5,  wf1_lr_5  = clf_cv_metrics(lr, X_clf_scaled, y_clf, skf)
acc_lr_lo, mf1_lr_lo, wf1_lr_lo = clf_cv_metrics(lr, X_clf_scaled, y_clf, loocv)

# ─── Random Forest Classifier ─────────────────────────────
rfc = RandomForestClassifier(n_estimators=100, max_depth=5,
                              class_weight="balanced", random_state=42)
acc_rf_5,  mf1_rf_5,  wf1_rf_5  = clf_cv_metrics(rfc, X_clf,       y_clf, skf)
acc_rf_lo, mf1_rf_lo, wf1_rf_lo = clf_cv_metrics(rfc, X_clf,       y_clf, loocv)

# ─── Таблица 5 ────────────────────────────────────────────
print("\n" + "═"*60)
print("  ТАБЛИЦА 5 — Метрики классификации (5-fold CV / LOOCV)")
print("═"*60)
print(f"  {'Алгоритм':<22} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
print("  " + "-"*56)
print(f"  {'Decision Tree (5-fold)':<22} {acc_dt_5:>10.4f} {mf1_dt_5:>10.4f} {wf1_dt_5:>12.4f}")
print(f"  {'Decision Tree (LOOCV)':<22} {acc_dt_lo:>10.4f} {mf1_dt_lo:>10.4f} {wf1_dt_lo:>12.4f}")
print(f"  {'LogReg (5-fold)':<22} {acc_lr_5:>10.4f} {mf1_lr_5:>10.4f} {wf1_lr_5:>12.4f}")
print(f"  {'LogReg (LOOCV)':<22} {acc_lr_lo:>10.4f} {mf1_lr_lo:>10.4f} {wf1_lr_lo:>12.4f}")
print(f"  {'Random Forest (5-fold)':<22} {acc_rf_5:>10.4f} {mf1_rf_5:>10.4f} {wf1_rf_5:>12.4f}")
print(f"  {'Random Forest (LOOCV)':<22} {acc_rf_lo:>10.4f} {mf1_rf_lo:>10.4f} {wf1_rf_lo:>12.4f}")
print()


# ─── Рис. 7 — Дерево решений ──────────────────────────────
dt.fit(X_clf, y_clf)

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    dt,
    feature_names=feat_cols_clf,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    impurity=True,
    proportion=False,
)
ax.set_title("Рис. 7 — Дерево решений: классификация уровня девиантного риска (max_depth=4)",
             fontsize=13)
savefig("fig7_decision_tree.png")


# ─── Лучшая модель для матрицы ошибок ─────────────────────
# Определяем лучшую по 5-fold Macro F1
scores_5fold = {
    "Decision Tree":       mf1_dt_5,
    "Logistic Regression": mf1_lr_5,
    "Random Forest":       mf1_rf_5,
}
best_clf_name = max(scores_5fold, key=scores_5fold.get)
print(f"  Лучшая классификационная модель (Macro F1): {best_clf_name}")

# Обучаем лучшую и предсказываем (на всей выборке для CM)
if best_clf_name == "Decision Tree":
    best_clf = dt
    X_best   = X_clf
elif best_clf_name == "Logistic Regression":
    best_clf = lr
    X_best   = X_clf_scaled
else:
    best_clf = rfc
    X_best   = X_clf

best_clf.fit(X_best, y_clf)
y_pred_best = best_clf.predict(X_best)

# ─── Рис. 8 — Матрица ошибок ─────────────────────────────
cm = confusion_matrix(y_clf, y_pred_best)

fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Рис. 8 — Матрица ошибок: {best_clf_name}")
savefig("fig8_confusion_matrix.png")


# ─── Рис. 9 — Важность признаков RF Classifier ───────────
rfc.fit(X_clf, y_clf)
fi = pd.Series(rfc.feature_importances_, index=feat_cols_clf)
fi_top10 = fi.nlargest(10)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(fi_top10.index[::-1], fi_top10.values[::-1],
               color="#70AD47", alpha=0.85)
ax.set_xlabel("Gini importance")
ax.set_title("Рис. 9 — Важность признаков: Random Forest Classifier")
for bar, val in zip(bars, fi_top10.values[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
savefig("fig9_feature_importance_clf.png")


# ─── Таблица 6 — Топ-10 признаков классификатора ──────────
print("═"*60)
print("  ТАБЛИЦА 6 — Топ-10 признаков (Random Forest Classifier)")
print("═"*60)
print(f"  {'Ранг':>4}  {'Признак':<45} {'Gini importance':>16}")
print("  " + "-"*68)
for rank, (feat, val) in enumerate(fi.nlargest(10).items(), 1):
    print(f"  {rank:>4}  {feat:<45} {val:>16.4f}")
print()


# ─── Classification Report ────────────────────────────────
print("─"*60)
print(f"  Classification Report ({best_clf_name}, обучение на всей выборке):")
print("─"*60)
print(classification_report(y_clf, y_pred_best, target_names=class_names))

print("\n" + "═"*60)
print("  ВСЕ РИСУНКИ И МЕТРИКИ ГОТОВЫ")
print("═"*60)