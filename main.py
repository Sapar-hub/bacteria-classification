import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, balanced_accuracy_score

# Интерполяция
def interpolate_common_grid(X, y, wavs, step=1.0):
    def get_range(cls):
        mask = y == cls
        data = X[mask]
        valid = (~np.isnan(data)).any(axis=0)
        return wavs[valid].min(), wavs[valid].max()
    r0, r1 = get_range(0), get_range(1)
    start, end = max(r0[0], r1[0]), min(r0[1], r1[1])
    grid = np.arange(start, end + step, step)
    return np.array([
        interp1d(wavs[~np.isnan(row)], row[~np.isnan(row)],
                 kind='nearest', bounds_error=False, fill_value=0)(grid)
        for row in X
    ]), grid


# Извлечение признаков
def extract_peak_features(X, wvs, top_n=10, prominence=0.02, distance=2):
    features = []
    for row in X:
        peaks, _ = find_peaks(row, prominence=prominence, distance=distance)
        heights = row[peaks]
        positions = wvs[peaks]
        widths, *_ = peak_widths(row, peaks, rel_height=0.5)
        if len(peaks) >= top_n:
            idxs = np.argsort(heights)[-top_n:]
        else:
            idxs = np.arange(len(peaks))
        feat = []
        for i in idxs[::-1]:
            feat.extend([positions[i], heights[i], widths[i]])
        while len(feat) < top_n*3:
            feat.extend([0,0,0])
        features.append(feat)
    return np.array(features)

# Загрузка датасета
df = pd.read_csv("FINAL.csv")
meta_cols = {"Unnamed: 0", "spectrumID", "Class"}

wavs = np.array(sorted([float(c) for c in df.columns if c not in meta_cols]))
X_raw = df[[str(w) for w in wavs]].values
y = df["Class"].values
groups = df["spectrumID"].values

X_interp, common_wavs = interpolate_common_grid(X_raw, y, wavs)

# Биннинг
n_bins = 20
bin_size = len(common_wavs) // n_bins
binned_wavs = np.array([common_wavs[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)])
X_binned = np.array([
    [row[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)]
    for row in X_interp
])

# MinMax нормирование
X_scaled = np.array([MinMaxScaler().fit_transform(row.reshape(-1,1)).flatten() for row in X_binned])
X_features = extract_peak_features(X_scaled, binned_wavs)


# 1. Составим пайплайн: (опционально) масштабирование → классификатор
pipeline = Pipeline([
    ("scaler", MinMaxScaler()),  # можно заменить или убрать
    ("clf", RandomForestClassifier(random_state=42))
])

# 2. Задаём сетку гиперпараметров
param_grid = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__class_weight": [None, "balanced"]
}

# 3. Настраиваем GroupKFold по spectrumID
gkf = GroupKFold(n_splits=len(np.unique(groups)))

# 4. Создаём GridSearchCV с метрикой сбалансированной точности
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=gkf.split(X_features, y, groups=groups),
    scoring=make_scorer(balanced_accuracy_score),
    n_jobs=-1,
    verbose=2
)

# 5. Запускаем подбор
grid.fit(X_features, y)

# 6. Смотрим результаты
print("Лучшие параметры:", grid.best_params_)
print("Лучшее значение balanced_accuracy:", grid.best_score_)
