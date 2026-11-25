import os
import warnings
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore')




# 1. WCZYTANIE DANYCH I WSTĘPNA WERYFIKACJA STRUKTURY

SCIEZKA_PLIKU = 'bank-additional-full.csv'
df = pd.read_csv(SCIEZKA_PLIKU, sep=';')
   

# Kontrola integralności danych - wyświetlenie skrajnych obserwacji
print(df.head(10))
print(df.tail(10))

# Analiza typów zmiennych oraz obciążenia pamięci RAM
df.info()


# 2. KONTROLA BRAKÓW I ANOMALII DANYCH

braki_danych = df.isnull().sum().to_frame(name='Liczba braków')
braki_danych['Udział %'] = (braki_danych['Liczba braków'] / len(df)) * 100
print(braki_danych)

kolumny_tekstowe = df.select_dtypes(include=['object']).columns
for kolumna in kolumny_tekstowe:
    liczba_unknown = (df[kolumna] == 'unknown').sum()
    if liczba_unknown > 0:
        udzial_unknown = (liczba_unknown / len(df)) * 100
        print(f"Zmienna '{kolumna}': {liczba_unknown} wartości 'unknown' ({udzial_unknown:.2f}%)")


# 3. STATYSTYKI OPISOWE ZMIENNYCH NUMERYCZNYCH

kolumny_numeryczne = df.select_dtypes(include=['int64', 'float64']).columns

# Wyznaczenie miar położenia, rozproszenia oraz asymetrii (skośności)
statystyki_numeryczne = df.describe().T
statystyki_numeryczne['skewness'] = df[kolumny_numeryczne].skew()

# Zapis raportu do pliku Excel
statystyki_numeryczne.to_excel('statystyki_opisowe_numeryczne.xlsx', sheet_name='Statystyki Opisowe')
print(statystyki_numeryczne)


# 4. STATYSTYKI OPISOWE ZMIENNYCH KATEGORYCZNYCH (TABELE CZĘSTOŚCI)


with pd.ExcelWriter('statystyki_jakosciowe.xlsx', engine='openpyxl') as writer:
    for kolumna in kolumny_tekstowe:
        czestosc_bezwzgledna = df[kolumna].value_counts()
        tabela_czestosci = pd.DataFrame({
            'Liczba wystąpień': czestosc_bezwzgledna,
            'Udział %': (czestosc_bezwzgledna / czestosc_bezwzgledna.sum()) * 100
        })
        
        nazwa_arkusza = f"Częstość - {kolumna}"[:31]
        tabela_czestosci.to_excel(writer, sheet_name=nazwa_arkusza)
        
        print(f"\nRozkład zmiennej kategorycznej: {kolumna}")
        print(tabela_czestosci)
        print("-" * 50)


# 5. WIZUALIZACJA: HISTOGRAMY ROZKŁADÓW ZMIENNYCH NUMERYCZNYCH
# 

FOLDER_NUMERYCZNE = 'wykresy_do_raportu_numeryczne'
os.makedirs(FOLDER_NUMERYCZNE, exist_ok=True)


for kolumna in kolumny_numeryczne:
    plt.figure(figsize=(7, 4))
    
    # Wykorzystanie zaawansowanego modułu histplot z biblioteki Seaborn
    sns.histplot(df[kolumna], kde=False, color='royalblue', edgecolor='black', bins=30)
    
    plt.title(f'Rozkład zmiennej: {kolumna}', fontweight='bold', pad=10)
    plt.xlabel(kolumna)
    plt.ylabel('Liczba obserwacji')
    
    plt.tight_layout()
    sciezka_wykresu = os.path.join(FOLDER_NUMERYCZNE, f"histogram_{kolumna}.png")
    plt.savefig(sciezka_wykresu, dpi=300)
    plt.close()


# 6. WIZUALIZACJA: ROZKŁADY ZMIENNYCH KATEGORYCZNYCH

FOLDER_KATEGORYCZNE = 'wykresy_do_raportu_kategoryczne'
os.makedirs(FOLDER_KATEGORYCZNE, exist_ok=True)


for kolumna in kolumny_tekstowe:
    plt.figure(figsize=(7, 4))
    
    czestosc = df[kolumna].value_counts()
    
    sns.barplot(x=czestosc.values, y=czestosc.index, hue=czestosc.index, palette='viridis', edgecolor='black', legend=False)
    
    plt.title(f'Rozkład zmiennej: {kolumna}', fontweight='bold', pad=10)
    plt.xlabel('Liczba wystąpień')
    plt.ylabel(kolumna)
    
    plt.tight_layout()
    sciezka_wykresu = os.path.join(FOLDER_KATEGORYCZNE, f"wykres_slupkowy_{kolumna}.png")
    plt.savefig(sciezka_wykresu, dpi=300)
    plt.close()


# 7. ANALIZA KORELACJI ZMIENNYCH NUMERYCZNYCH (PARAMETRYCZNA)

macierz_korelacji = df[kolumny_numeryczne].corr(method='pearson')

macierz_korelacji.to_excel('macierz_korelacji.xlsx', sheet_name='Korelacja Pearsona')

# Generowanie czytelnej mapy ciepła (Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(
    macierz_korelacji, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    vmin=-1, 
    vmax=1, 
    linewidths=0.5,
    cbar_kws={"shrink": .8}
)
plt.title('Mapa ciepła korelacji zmiennych numerycznych', fontweight='bold', pad=15)
plt.tight_layout()

sciezka_heatmapy = os.path.join(FOLDER_NUMERYCZNE, 'macierz_korelacji_heatmap.png')
plt.savefig(sciezka_heatmapy, dpi=300)
plt.close()


# 8. ANALIZA ZALEŻNOŚCI STOCHASTYCZNYCH (TESTY CHI-KWADRAT)


wyniki_testow_chi2 = []

for zm1, zm2 in combinations(kolumny_tekstowe, 2):
    tabela_kontyngencji = pd.crosstab(df[zm1], df[zm2])
    
    if tabela_kontyngencji.size == 0:
        continue
        
    chi2, p_value, dof, expected = chi2_contingency(tabela_kontyngencji)
    Status_istotnosci = "ISTOTNA ZALEŻNOŚĆ" if p_value < 0.05 else "BRAK PODSTAW DO ODRZUCENIA H0"
    
    wyniki_testow_chi2.append({
        'Zmienna A': zm1,
        'Zmienna B': zm2,
        'Statystyka Chi2': round(chi2, 4),
        'p-value': p_value,  
        'Wynik testu (alpha=0.05)': Status_istotnosci
    })

raport_chi2 = pd.DataFrame(wyniki_testow_chi2)

raport_chi2 = raport_chi2.sort_values(by='p-value').reset_index(drop=True)

raport_chi2['p-value'] = raport_chi2['p-value'].apply(lambda x: f"{x:.4e}" if x < 0.0001 else round(x, 4))

print(raport_chi2.to_string(index=False))
raport_chi2.to_excel('raport_testow_chi_kwadrat.xlsx', index=False, sheet_name='Testy Chi2')

