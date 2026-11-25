import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_curve, roc_auc_score, 
                             precision_score, recall_score, f1_score, 
                             confusion_matrix, brier_score_loss)

import warnings
warnings.filterwarnings('ignore')


# ETAP 2: DEFINICJE FUNKCJI POMOCNICZYCH

def check_linearity(X, y, var, save_path=None):
    """Sprawdza założenie o liniowości względem logitu dla zmiennych ciągłych."""
    temp_df = X.copy()
    temp_df['target'] = y
    
    # Podział na decyle
    temp_df['bin'] = pd.qcut(temp_df[var], q=10, duplicates='drop')
    grouped = temp_df.groupby('bin', observed=True)['target'].mean()
    
    # Przekształcenie na logit z zabezpieczeniem przed log(0)
    epsilon = 1e-10
    logit = np.log((grouped + epsilon) / (1 - grouped + epsilon))
    
    plt.figure(figsize=(8, 4))
    plt.plot(logit.values, marker='o')
    plt.title(f'Weryfikacja liniowości dla: {var}')
    plt.xlabel('Decyle')
    plt.ylabel('Logit')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'linearity_{var}.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_advanced_metrics(y_true, y_prob, threshold=0.5):
    """Oblicza zaawansowane metryki dyskryminacyjne i kalibracyjne modelu."""
    y_pred = (y_prob >= threshold).astype(int) 
    
    # 1. Odsetek błędnych klasyfikacji (MCR)
    mcr = 1 - accuracy_score(y_true, y_pred)
    # 2. Average Squared Error (ASE / Brier Score)
    ase = brier_score_loss(y_true, y_prob)
    # 3. GINI
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    # 4. Kolmogorov-Smirnov (KS)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    
    return {"MCR": mcr, "ASE": ase, "GINI": gini, "KS": ks}

def evaluate_model_performance(y_true, y_prob, threshold, dataset_name):
    """Wypisuje kompleksowy raport z wynikami dla danego zbioru."""
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"--- WYNIKI DLA ZBIORU: {dataset_name.upper()} ---")
    print(f"Próg odcięcia:         {threshold:.4f}")
    print(f"Dokładność (Accuracy): {acc:.4f}")
    print(f"ROC AUC:               {auc:.4f}")
    print(f"Precyzja (Precision):  {prec:.4f}")
    print(f"Czułość (Recall):      {rec:.4f}")
    print(f"F1-Score:              {f1:.4f}\n")


# ETAP 3: WCZYTANIE I PODZIAŁ DANYCH

print("Ładowanie danych...")
bank_marketing = pd.read_csv('bank_marketing_split.csv')

# Tworzenie zbiorów na podstawie kolumny 'split'
train_df = bank_marketing[bank_marketing['split'] == 'TRAIN'].copy()
valid_df = bank_marketing[bank_marketing['split'] == 'VALID'].copy()
test_df  = bank_marketing[bank_marketing['split'] == 'TEST'].copy()

# Separacja cech i zmiennej docelowej
X_train, y_train = train_df.drop(columns=['split', 'y']), train_df['y']
X_valid, y_valid = valid_df.drop(columns=['split', 'y']), valid_df['y']
X_test,  y_test  = test_df.drop(columns=['split', 'y']),  test_df['y']

print(f"Kształty zbiorów -> Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

# Mapowanie zmiennej objaśnianej (Klasyfikacja binarna)
y_train = y_train.map({'yes': 1, 'no': 0})
y_valid = y_valid.map({'yes': 1, 'no': 0})
y_test  = y_test.map({'yes': 1, 'no': 0})


# ETAP 4: INŻYNIERIA CECH 

print("\nRozpoczęcie inżynierii cech...")
cat_cols = X_train.select_dtypes(include=['object', "string"]).columns.tolist()
num_cols = X_train.select_dtypes(exclude=['object', "string"]).columns.tolist()

# One-Hot Encoding dla zmiennych kategorialnych (z drop_first dla uniknięcia współliniowości)
X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_valid_enc = pd.get_dummies(X_valid, columns=cat_cols, drop_first=True)
X_test_enc  = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# Wyrównanie kolumn (obsługa brakujących kategorii w mniejszych zbiorach)
X_valid_enc = X_valid_enc.reindex(columns=X_train_enc.columns, fill_value=0)
X_test_enc  = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

# Generowanie wykresów liniowości 
nazwa_folderu = 'wykresy_do_raportu_liniowosc'
os.makedirs(nazwa_folderu, exist_ok=True)
for col in num_cols:
    check_linearity(X_train, y_train, col, save_path=nazwa_folderu)

# Binowanie zmiennych niespełniających założenia o liniowości
continuous_vars = ['age', 'pdays','campaign', 'emp.var.rate', 'cons.price.idx', 
                   'cons.conf.idx','euribor3m', 'nr.employed']

all_bins = {}
for var in continuous_vars:
    # Wyznaczanie granic koszyków na zbiorze uczącym (kwartyle)
    _, bins = pd.qcut(X_train_enc[var], q=4, retbins=True, duplicates='drop')
    bins[0], bins[-1] = -np.inf, np.inf
    all_bins[var] = bins
    
    # Aplikacja koszyków do wszystkich zbiorów
    X_train_enc[f'{var}_bin'] = pd.cut(X_train_enc[var], bins=bins, labels=False)
    X_valid_enc[f'{var}_bin'] = pd.cut(X_valid_enc[var], bins=bins, labels=False)
    X_test_enc[f'{var}_bin']  = pd.cut(X_test_enc[var],  bins=bins, labels=False)

# OHE dla nowych zmiennych zbinowanych
cols_to_encode = [f'{var}_bin' for var in continuous_vars]

X_train_final = pd.get_dummies(X_train_enc.drop(columns=continuous_vars), columns=cols_to_encode, drop_first=True)
X_valid_final = pd.get_dummies(X_valid_enc.drop(columns=continuous_vars), columns=cols_to_encode, drop_first=True)
X_test_final  = pd.get_dummies(X_test_enc.drop(columns=continuous_vars),  columns=cols_to_encode, drop_first=True)

# Ostateczne wyrównanie wymiarów
X_valid_final = X_valid_final.reindex(columns=X_train_final.columns, fill_value=0)
X_test_final  = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)


# ETAP 5: TRENOWANIE MODELU I STATYSTYKI

print("\nTrenowanie modelu regresji logistycznej...")
# max_iter=1000 by zapewnić zbieżność optymalizatora
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_final, y_train)

# Estymacja modelu za pomocą statsmodels
X_train_stat = sm.add_constant(X_train_final.astype(float))
y_train_stat = y_train.astype(float)

try:
    modelRL = sm.Logit(y_train_stat, X_train_stat)
    result_LR = modelRL.fit(disp=False)
    print("\n--- Podsumowanie modelu z biblioteki Statsmodels ---")
    print(result_LR.summary2()) 
except Exception as e:
    print(f"Ostrzeżenie: Nie udało się dopasować modelu statsmodels (częste przy idealnej separacji). Błąd: {e}")

# Wyznaczanie kryteriów informacyjnych AIC / BIC ręcznie dla modelu sklearn
p_train = logreg.predict_proba(X_train_final)[:, 1]
LL = np.sum(y_train * np.log(p_train + 1e-10) + (1 - y_train) * np.log(1 - p_train + 1e-10))
k = len(logreg.coef_[0]) + 1  
n = len(y_train)

print(f"\nKryteria informacyjne dla modelu Sklearn:")
print(f"AIC: {2*k - 2*LL:.2f}")
print(f"BIC: {np.log(n)*k - 2*LL:.2f}\n")


# ETAP 6: EWALUACJA I OPTYMALIZACJA PROGU

# Prognozowanie prawdopodobieństw dla wszystkich zbiorów
y_prob_train = logreg.predict_proba(X_train_final)[:, 1]
y_prob_valid = logreg.predict_proba(X_valid_final)[:, 1]
y_prob_test  = logreg.predict_proba(X_test_final)[:, 1]

# Poszukiwanie optymalnego punktu odcięcia (Kryterium Youdena) na zbiorze walidacyjnym
fpr, tpr, thresholds = roc_curve(y_valid, y_prob_valid)
youden_j = tpr - fpr
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]

print(f"Optymalny próg odcięcia (Youden's J wyznaczony na Valid): {optimal_threshold:.4f}\n")

# Raport z klasycznymi metrykami dla każdego zbioru
evaluate_model_performance(y_train, y_prob_train, optimal_threshold, "Train")
evaluate_model_performance(y_valid, y_prob_valid, optimal_threshold, "Validation")
evaluate_model_performance(y_test,  y_prob_test,  optimal_threshold, "Test")

# Obliczenie zaawansowanych metryk analitycznych
metrics_train = calculate_advanced_metrics(y_train, y_prob_train, optimal_threshold)
metrics_valid = calculate_advanced_metrics(y_valid, y_prob_valid, optimal_threshold)
metrics_test  = calculate_advanced_metrics(y_test,  y_prob_test,  optimal_threshold)

results_df = pd.DataFrame(
    [metrics_train, metrics_valid, metrics_test], 
    index=['Train', 'Valid', 'Test']
)

print("--- ZESTAWIENIE ZAAWANSOWANYCH METRYK (Zoptymalizowany próg) ---")
print(results_df.round(4))


# ETAP 7: WIZUALIZACJA ZBIORCZA (ROC & MACIERZ POMYŁEK)


# Rysowanie Krzywej ROC dla zbioru walidacyjnego
plt.figure(figsize=(7, 6))
auc_valid = roc_auc_score(y_valid, y_prob_valid)
plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC curve Valid (AUC = {auc_valid:.3f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Klasyfikator losowy')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='black', label='Youden Index', s=100)

plt.title('Krzywa ROC - Zbiór Walidacyjny', fontsize=14)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="lower right")
plt.show()

# Macierz pomyłek dla zbioru testowego z wykorzystaniem optymalnego progu
y_pred_test_opt = (y_prob_test >= optimal_threshold).astype(int)
cm_final_test = confusion_matrix(y_test, y_pred_test_opt)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_final_test, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Predykcja: 0', 'Predykcja: 1'],
            yticklabels=['Rzeczywistość: 0', 'Rzeczywistość: 1'])
plt.title(f"Macierz pomyłek – Zbiór Testowy (Próg: {optimal_threshold:.2f})")
plt.show()
