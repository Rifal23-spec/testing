import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Data penelitian
data = {
    'Responden': range(1, 21),
    'Pelayanan': [21, 21, 23, 19, 20, 19, 21, 19, 21, 20,
                  22, 21, 20, 18, 17, 20, 22, 21, 20, 18],
    'Kepuasan_Pelanggan': [20, 21, 22, 20, 21, 22, 23, 21, 22, 21,
                           23, 22, 20, 19, 18, 21, 22, 21, 22, 20]
}

df = pd.DataFrame(data)

# Variabel X (Pelayanan) dan Y (Kepuasan Pelanggan)
X = df['Pelayanan'].values
Y = df['Kepuasan_Pelanggan'].values
n = len(X)

print("="*80)
print("ANALISIS REGRESI LINEAR SEDERHANA")
print("Pengaruh Pelayanan (X) terhadap Kepuasan Pelanggan (Y)")
print("="*80)
print("\nData:")
print(df.to_string(index=False))
print(f"\nJumlah Responden (n) = {n}")

# Perhitungan statistik dasar
sum_X = np.sum(X)
sum_Y = np.sum(Y)
sum_X2 = np.sum(X**2)
sum_Y2 = np.sum(Y**2)
sum_XY = np.sum(X*Y)
mean_X = np.mean(X)
mean_Y = np.mean(Y)

print(f"\nΣX = {sum_X}")
print(f"ΣY = {sum_Y}")
print(f"ΣX² = {sum_X2}")
print(f"ΣY² = {sum_Y2}")
print(f"ΣXY = {sum_XY}")
print(f"Mean X = {mean_X:.4f}")
print(f"Mean Y = {mean_Y:.4f}")

print("\n" + "="*80)
print("1. PERSAMAAN REGRESI")
print("="*80)

# Menghitung koefisien regresi b1 dan b0
# b1 = (n*ΣXY - ΣX*ΣY) / (n*ΣX² - (ΣX)²)
b1 = (n * sum_XY - sum_X * sum_Y) / (n * sum_X2 - sum_X**2)

# b0 = Ȳ - b1*X̄
b0 = mean_Y - b1 * mean_X

print(f"\nKoefisien Regresi (b1/slope) = {b1:.6f}")
print(f"Konstanta (b0/intercept) = {b0:.6f}")
print(f"\nPersamaan Regresi:")
print(f"Ŷ = {b0:.6f} + {b1:.6f}X")
print(f"atau")
print(f"Kepuasan Pelanggan = {b0:.4f} + {b1:.4f}(Pelayanan)")

# Interpretasi
print(f"\nInterpretasi:")
print(f"- Jika pelayanan = 0, maka kepuasan pelanggan = {b0:.4f}")
print(f"- Setiap kenaikan 1 poin pelayanan, kepuasan pelanggan naik {b1:.4f} poin")

print("\n" + "="*80)
print("2. KOEFISIEN DETERMINASI (R²)")
print("="*80)

# Menghitung Y prediksi
Y_pred = b0 + b1 * X

# Sum of Squares
SS_total = np.sum((Y - mean_Y)**2)  # Total Sum of Squares
SS_regression = np.sum((Y_pred - mean_Y)**2)  # Regression Sum of Squares
SS_residual = np.sum((Y - Y_pred)**2)  # Residual Sum of Squares

# Koefisien Determinasi (R²)
R_squared = SS_regression / SS_total

# Koefisien Korelasi (r)
r = np.sqrt(R_squared) if b1 > 0 else -np.sqrt(R_squared)

print(f"\nSS Total (SST) = {SS_total:.6f}")
print(f"SS Regression (SSR) = {SS_regression:.6f}")
print(f"SS Residual (SSE) = {SS_residual:.6f}")
print(f"\nKoefisien Korelasi (r) = {r:.6f}")
print(f"Koefisien Determinasi (R²) = {R_squared:.6f}")
print(f"Koefisien Determinasi (R²) dalam % = {R_squared*100:.2f}%")

print(f"\nInterpretasi:")
print(f"- {R_squared*100:.2f}% variasi kepuasan pelanggan dapat dijelaskan oleh pelayanan")
print(f"- {(1-R_squared)*100:.2f}% dijelaskan oleh faktor lain di luar model")

print("\n" + "="*80)
print("3. KESALAHAN ESTIMASI BAKU (Standard Error of Estimate)")
print("="*80)

# Standard Error of Estimate (Se)
Se = np.sqrt(SS_residual / (n - 2))

print(f"\nStandard Error of Estimate (Se) = {Se:.6f}")
print(f"\nInterpretasi:")
print(f"Rata-rata kesalahan prediksi model adalah {Se:.4f} poin")

print("\n" + "="*80)
print("4. STANDAR ERROR KOEFISIEN REGRESI")
print("="*80)

# Standard Error of Regression Coefficient (Sb1)
Sb1 = Se / np.sqrt(np.sum((X - mean_X)**2))

# Standard Error of Constant (Sb0)
Sb0 = Se * np.sqrt((1/n) + (mean_X**2 / np.sum((X - mean_X)**2)))

print(f"\nStandard Error koefisien b1 (Sb1) = {Sb1:.6f}")
print(f"Standard Error konstanta b0 (Sb0) = {Sb0:.6f}")

print("\n" + "="*80)
print("5. UJI F (Uji Signifikansi Model)")
print("="*80)

# Uji F
df_regression = 1  # derajat bebas regresi
df_residual = n - 2  # derajat bebas residual
MS_regression = SS_regression / df_regression  # Mean Square Regression
MS_residual = SS_residual / df_residual  # Mean Square Residual
F_hitung = MS_regression / MS_residual

# F tabel dengan alpha = 0.05
alpha = 0.05
F_tabel = stats.f.ppf(1 - alpha, df_regression, df_residual)

# P-value
p_value_F = 1 - stats.f.cdf(F_hitung, df_regression, df_residual)

print(f"\nTabel ANOVA:")
print(f"{'Sumber Variasi':<20} {'df':<8} {'SS':<15} {'MS':<15} {'F':<12}")
print("-" * 70)
print(f"{'Regresi':<20} {df_regression:<8} {SS_regression:<15.6f} {MS_regression:<15.6f} {F_hitung:<12.6f}")
print(f"{'Residual':<20} {df_residual:<8} {SS_residual:<15.6f} {MS_residual:<15.6f}")
print(f"{'Total':<20} {n-1:<8} {SS_total:<15.6f}")

print(f"\nF hitung = {F_hitung:.6f}")
print(f"F tabel (α=0.05, df1={df_regression}, df2={df_residual}) = {F_tabel:.6f}")
print(f"P-value = {p_value_F:.6f}")

print(f"\nHipotesis:")
print(f"H0: Model regresi tidak signifikan (β1 = 0)")
print(f"H1: Model regresi signifikan (β1 ≠ 0)")
print(f"\nKriteria Pengujian:")
print(f"Jika F hitung > F tabel, maka H0 ditolak")
print(f"Jika P-value < α (0.05), maka H0 ditolak")

print(f"\nKesimpulan:")
if F_hitung > F_tabel:
    print(f"F hitung ({F_hitung:.4f}) > F tabel ({F_tabel:.4f})")
    print(f"P-value ({p_value_F:.6f}) < α (0.05)")
    print(f"KESIMPULAN: H0 DITOLAK")
    print(f"Model regresi SIGNIFIKAN pada tingkat kepercayaan 95%")
    print(f"Pelayanan secara statistik berpengaruh signifikan terhadap Kepuasan Pelanggan")
else:
    print(f"F hitung ({F_hitung:.4f}) ≤ F tabel ({F_tabel:.4f})")
    print(f"P-value ({p_value_F:.6f}) ≥ α (0.05)")
    print(f"KESIMPULAN: H0 DITERIMA")
    print(f"Model regresi TIDAK SIGNIFIKAN")
    print(f"Pelayanan tidak berpengaruh signifikan terhadap Kepuasan Pelanggan")

print("\n" + "="*80)
print("6. UJI T (Uji Signifikansi Koefisien Regresi)")
print("="*80)

# Uji t untuk b1
t_hitung = b1 / Sb1

# t tabel (two-tailed)
t_tabel = stats.t.ppf(1 - alpha/2, df_residual)

# P-value (two-tailed)
p_value_t = 2 * (1 - stats.t.cdf(abs(t_hitung), df_residual))

print(f"\nUji t untuk Koefisien Regresi (b1):")
print(f"t hitung = b1 / Sb1 = {b1:.6f} / {Sb1:.6f} = {t_hitung:.6f}")
print(f"t tabel (α=0.05, df={df_residual}, two-tailed) = ±{t_tabel:.6f}")
print(f"P-value = {p_value_t:.6f}")

print(f"\nHipotesis:")
print(f"H0: Pelayanan tidak berpengaruh signifikan terhadap Kepuasan Pelanggan (β1 = 0)")
print(f"H1: Pelayanan berpengaruh signifikan terhadap Kepuasan Pelanggan (β1 ≠ 0)")
print(f"\nKriteria Pengujian:")
print(f"Jika |t hitung| > t tabel, maka H0 ditolak")
print(f"Jika P-value < α (0.05), maka H0 ditolak")

print(f"\nKesimpulan:")
if abs(t_hitung) > t_tabel:
    print(f"|t hitung| ({abs(t_hitung):.4f}) > t tabel ({t_tabel:.4f})")
    print(f"P-value ({p_value_t:.6f}) < α (0.05)")
    print(f"KESIMPULAN: H0 DITOLAK")
    print(f"Koefisien regresi SIGNIFIKAN pada tingkat kepercayaan 95%")
    print(f"Pelayanan berpengaruh SIGNIFIKAN terhadap Kepuasan Pelanggan")
else:
    print(f"|t hitung| ({abs(t_hitung):.4f}) ≤ t tabel ({t_tabel:.4f})")
    print(f"P-value ({p_value_t:.6f}) ≥ α (0.05)")
    print(f"KESIMPULAN: H0 DITERIMA")
    print(f"Koefisien regresi TIDAK SIGNIFIKAN")
    print(f"Pelayanan tidak berpengaruh signifikan terhadap Kepuasan Pelanggan")

# Uji t untuk b0 (opsional)
t_hitung_b0 = b0 / Sb0
p_value_t_b0 = 2 * (1 - stats.t.cdf(abs(t_hitung_b0), df_residual))

print(f"\n\nUji t untuk Konstanta (b0) - Opsional:")
print(f"t hitung = b0 / Sb0 = {b0:.6f} / {Sb0:.6f} = {t_hitung_b0:.6f}")
print(f"P-value = {p_value_t_b0:.6f}")

print("\n" + "="*80)
print("RINGKASAN HASIL ANALISIS")
print("="*80)

print(f"\n1. Persamaan Regresi: Ŷ = {b0:.4f} + {b1:.4f}X")
print(f"2. Koefisien Determinasi (R²) = {R_squared:.4f} atau {R_squared*100:.2f}%")
print(f"3. Standard Error of Estimate (Se) = {Se:.4f}")
print(f"4. Standard Error Koefisien (Sb1) = {Sb1:.6f}")
print(f"5. Uji F: F hitung = {F_hitung:.4f} > F tabel = {F_tabel:.4f}")
print(f"   Kesimpulan: Model regresi {'SIGNIFIKAN' if F_hitung > F_tabel else 'TIDAK SIGNIFIKAN'}")
print(f"6. Uji t: t hitung = {t_hitung:.4f}, t tabel = ±{t_tabel:.4f}")
print(f"   Kesimpulan: Pelayanan berpengaruh {'SIGNIFIKAN' if abs(t_hitung) > t_tabel else 'TIDAK SIGNIFIKAN'}")

print("\n" + "="*80)

# Membuat visualisasi
plt.figure(figsize=(12, 8))

# Plot 1: Scatter plot dengan garis regresi
plt.subplot(2, 2, 1)
plt.scatter(X, Y, color='blue', alpha=0.6, s=100, edgecolors='black', label='Data Observasi')
plt.plot(X, Y_pred, color='red', linewidth=2, label=f'Garis Regresi: Ŷ={b0:.2f}+{b1:.2f}X')
plt.xlabel('Pelayanan (X)', fontsize=11, fontweight='bold')
plt.ylabel('Kepuasan Pelanggan (Y)', fontsize=11, fontweight='bold')
plt.title('Analisis Regresi Linear\nPelayanan vs Kepuasan Pelanggan', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residual plot
plt.subplot(2, 2, 2)
residuals = Y - Y_pred
plt.scatter(Y_pred, residuals, color='green', alpha=0.6, s=100, edgecolors='black')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Nilai Prediksi (Ŷ)', fontsize=11, fontweight='bold')
plt.ylabel('Residual (Y - Ŷ)', fontsize=11, fontweight='bold')
plt.title('Plot Residual', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 3: Histogram residual
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=8, color='purple', alpha=0.7, edgecolor='black')
plt.xlabel('Residual', fontsize=11, fontweight='bold')
plt.ylabel('Frekuensi', fontsize=11, fontweight='bold')
plt.title('Distribusi Residual', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Plot 4: Q-Q plot
plt.subplot(2, 2, 4)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot (Normal Probability Plot)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/vercel/sandbox/hasil_analisis_regresi.png', dpi=300, bbox_inches='tight')
print("\nGrafik visualisasi berhasil disimpan sebagai 'hasil_analisis_regresi.png'")

# Membuat tabel prediksi
print("\n" + "="*80)
print("TABEL PREDIKSI DAN RESIDUAL")
print("="*80)

hasil = pd.DataFrame({
    'Responden': range(1, n+1),
    'X (Pelayanan)': X,
    'Y (Kepuasan)': Y,
    'Ŷ (Prediksi)': Y_pred.round(4),
    'Residual (e)': residuals.round(4),
    'e²': (residuals**2).round(4)
})

print("\n" + hasil.to_string(index=False))
print(f"\nΣe² = {np.sum(residuals**2):.4f} (SS Residual)")

print("\n" + "="*80)
print("ANALISIS SELESAI")
print("="*80)
