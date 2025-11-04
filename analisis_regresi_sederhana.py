import math

# Data penelitian
pelayanan = [21, 21, 23, 19, 20, 19, 21, 19, 21, 20,
             22, 21, 20, 18, 17, 20, 22, 21, 20, 18]
kepuasan = [20, 21, 22, 20, 21, 22, 23, 21, 22, 21,
            23, 22, 20, 19, 18, 21, 22, 21, 22, 20]

n = len(pelayanan)

print("="*80)
print("ANALISIS REGRESI LINEAR SEDERHANA")
print("Pengaruh Pelayanan (X) terhadap Kepuasan Pelanggan (Y)")
print("="*80)

print("\nData:")
print(f"{'Responden':<12} {'Pelayanan':<12} {'Kepuasan':<12}")
print("-" * 40)
for i in range(n):
    print(f"{i+1:<12} {pelayanan[i]:<12} {kepuasan[i]:<12}")

print(f"\nJumlah Responden (n) = {n}")

# Perhitungan statistik dasar
sum_X = sum(pelayanan)
sum_Y = sum(kepuasan)
sum_X2 = sum([x**2 for x in pelayanan])
sum_Y2 = sum([y**2 for y in kepuasan])
sum_XY = sum([pelayanan[i] * kepuasan[i] for i in range(n)])
mean_X = sum_X / n
mean_Y = sum_Y / n

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

print(f"\nRumus:")
print(f"b1 = (n*ΣXY - ΣX*ΣY) / (n*ΣX² - (ΣX)²)")
print(f"b1 = ({n}*{sum_XY} - {sum_X}*{sum_Y}) / ({n}*{sum_X2} - {sum_X}²)")
print(f"b1 = ({n * sum_XY} - {sum_X * sum_Y}) / ({n * sum_X2} - {sum_X**2})")
print(f"b1 = {n * sum_XY - sum_X * sum_Y} / {n * sum_X2 - sum_X**2}")
print(f"b1 = {b1:.6f}")

print(f"\nb0 = Ȳ - b1*X̄")
print(f"b0 = {mean_Y:.4f} - {b1:.6f}*{mean_X:.4f}")
print(f"b0 = {b0:.6f}")

print(f"\n{'='*40}")
print(f"PERSAMAAN REGRESI:")
print(f"Ŷ = {b0:.6f} + {b1:.6f}X")
print(f"atau")
print(f"Ŷ = {b0:.4f} + {b1:.4f}X")
print(f"{'='*40}")

print(f"\nInterpretasi:")
print(f"- Konstanta (b0) = {b0:.4f}")
print(f"  Jika pelayanan = 0, maka kepuasan pelanggan = {b0:.4f}")
print(f"- Koefisien Regresi (b1) = {b1:.4f}")
print(f"  Setiap kenaikan 1 poin pelayanan, kepuasan pelanggan naik {b1:.4f} poin")

print("\n" + "="*80)
print("2. KOEFISIEN DETERMINASI (R²)")
print("="*80)

# Menghitung Y prediksi
Y_pred = [b0 + b1 * x for x in pelayanan]

# Sum of Squares
SS_total = sum([(kepuasan[i] - mean_Y)**2 for i in range(n)])
SS_regression = sum([(Y_pred[i] - mean_Y)**2 for i in range(n)])
SS_residual = sum([(kepuasan[i] - Y_pred[i])**2 for i in range(n)])

# Koefisien Determinasi (R²)
R_squared = SS_regression / SS_total

# Koefisien Korelasi (r)
r = math.sqrt(R_squared) if b1 > 0 else -math.sqrt(R_squared)

print(f"\nPerhitungan Sum of Squares:")
print(f"SS Total (SST) = Σ(Y - Ȳ)² = {SS_total:.6f}")
print(f"SS Regression (SSR) = Σ(Ŷ - Ȳ)² = {SS_regression:.6f}")
print(f"SS Residual (SSE) = Σ(Y - Ŷ)² = {SS_residual:.6f}")

print(f"\nVerifikasi: SST = SSR + SSE")
print(f"{SS_total:.6f} = {SS_regression:.6f} + {SS_residual:.6f}")
print(f"{SS_total:.6f} = {SS_regression + SS_residual:.6f} ✓")

print(f"\nKoefisien Korelasi (r):")
print(f"r = √R² = √{R_squared:.6f} = {r:.6f}")

print(f"\n{'='*40}")
print(f"R² = SSR / SST")
print(f"R² = {SS_regression:.6f} / {SS_total:.6f}")
print(f"R² = {R_squared:.6f}")
print(f"R² = {R_squared*100:.2f}%")
print(f"{'='*40}")

print(f"\nInterpretasi:")
print(f"- Koefisien Determinasi (R²) = {R_squared:.4f} atau {R_squared*100:.2f}%")
print(f"- Artinya: {R_squared*100:.2f}% variasi kepuasan pelanggan dapat dijelaskan")
print(f"  oleh variabel pelayanan dalam model regresi ini")
print(f"- Sisanya {(1-R_squared)*100:.2f}% dijelaskan oleh faktor lain di luar model")

print("\n" + "="*80)
print("3. KESALAHAN ESTIMASI BAKU (Standard Error of Estimate)")
print("="*80)

# Standard Error of Estimate (Se)
Se = math.sqrt(SS_residual / (n - 2))

print(f"\nRumus:")
print(f"Se = √(SSE / (n - 2))")
print(f"Se = √({SS_residual:.6f} / ({n} - 2))")
print(f"Se = √({SS_residual:.6f} / {n - 2})")
print(f"Se = √{SS_residual / (n - 2):.6f}")

print(f"\n{'='*40}")
print(f"Standard Error of Estimate (Se) = {Se:.6f}")
print(f"{'='*40}")

print(f"\nInterpretasi:")
print(f"- Standard Error = {Se:.4f}")
print(f"- Rata-rata kesalahan prediksi model adalah {Se:.4f} poin")
print(f"- Semakin kecil nilai Se, semakin baik model dalam memprediksi")

print("\n" + "="*80)
print("4. STANDAR ERROR KOEFISIEN REGRESI")
print("="*80)

# Standard Error of Regression Coefficient (Sb1)
sum_x_minus_mean_squared = sum([(x - mean_X)**2 for x in pelayanan])
Sb1 = Se / math.sqrt(sum_x_minus_mean_squared)

# Standard Error of Constant (Sb0)
Sb0 = Se * math.sqrt((1/n) + (mean_X**2 / sum_x_minus_mean_squared))

print(f"\nPerhitungan Sb1 (Standard Error koefisien b1):")
print(f"Sb1 = Se / √Σ(X - X̄)²")
print(f"Sb1 = {Se:.6f} / √{sum_x_minus_mean_squared:.6f}")
print(f"Sb1 = {Se:.6f} / {math.sqrt(sum_x_minus_mean_squared):.6f}")
print(f"Sb1 = {Sb1:.6f}")

print(f"\nPerhitungan Sb0 (Standard Error konstanta b0):")
print(f"Sb0 = Se * √(1/n + X̄²/Σ(X - X̄)²)")
print(f"Sb0 = {Se:.6f} * √(1/{n} + {mean_X:.4f}²/{sum_x_minus_mean_squared:.6f})")
print(f"Sb0 = {Sb0:.6f}")

print(f"\n{'='*40}")
print(f"Standard Error koefisien b1 (Sb1) = {Sb1:.6f}")
print(f"Standard Error konstanta b0 (Sb0) = {Sb0:.6f}")
print(f"{'='*40}")

print("\n" + "="*80)
print("5. UJI F (Uji Signifikansi Model)")
print("="*80)

# Uji F
df_regression = 1  # derajat bebas regresi
df_residual = n - 2  # derajat bebas residual
MS_regression = SS_regression / df_regression
MS_residual = SS_residual / df_residual
F_hitung = MS_regression / MS_residual

# F tabel dengan alpha = 0.05, df1=1, df2=18
# Nilai F tabel dari tabel distribusi F
F_tabel = 4.41  # F(0.05, 1, 18)

print(f"\nTabel ANOVA:")
print(f"{'Sumber Variasi':<20} {'df':<8} {'SS':<15} {'MS':<15} {'F':<12}")
print("-" * 75)
print(f"{'Regresi':<20} {df_regression:<8} {SS_regression:<15.6f} {MS_regression:<15.6f} {F_hitung:<12.6f}")
print(f"{'Residual':<20} {df_residual:<8} {SS_residual:<15.6f} {MS_residual:<15.6f}")
print(f"{'Total':<20} {n-1:<8} {SS_total:<15.6f}")

print(f"\nPerhitungan F hitung:")
print(f"F = MS Regression / MS Residual")
print(f"F = {MS_regression:.6f} / {MS_residual:.6f}")
print(f"F = {F_hitung:.6f}")

print(f"\n{'='*40}")
print(f"F hitung = {F_hitung:.4f}")
print(f"F tabel (α=0.05, df1={df_regression}, df2={df_residual}) = {F_tabel:.2f}")
print(f"{'='*40}")

print(f"\nHipotesis:")
print(f"H0: Model regresi tidak signifikan (β1 = 0)")
print(f"    Pelayanan tidak berpengaruh terhadap Kepuasan Pelanggan")
print(f"H1: Model regresi signifikan (β1 ≠ 0)")
print(f"    Pelayanan berpengaruh terhadap Kepuasan Pelanggan")

print(f"\nTingkat Signifikansi: α = 0.05 (5%)")
print(f"Derajat Bebas: df1 = {df_regression}, df2 = {df_residual}")

print(f"\nKriteria Pengujian:")
print(f"Jika F hitung > F tabel, maka H0 ditolak (model signifikan)")
print(f"Jika F hitung ≤ F tabel, maka H0 diterima (model tidak signifikan)")

print(f"\n{'='*60}")
print(f"KESIMPULAN UJI F:")
if F_hitung > F_tabel:
    print(f"F hitung ({F_hitung:.4f}) > F tabel ({F_tabel:.2f})")
    print(f"H0 DITOLAK, H1 DITERIMA")
    print(f"\nModel regresi SIGNIFIKAN pada tingkat kepercayaan 95%")
    print(f"Variabel Pelayanan secara statistik berpengaruh SIGNIFIKAN")
    print(f"terhadap Kepuasan Pelanggan")
else:
    print(f"F hitung ({F_hitung:.4f}) ≤ F tabel ({F_tabel:.2f})")
    print(f"H0 DITERIMA")
    print(f"\nModel regresi TIDAK SIGNIFIKAN")
    print(f"Variabel Pelayanan tidak berpengaruh signifikan")
    print(f"terhadap Kepuasan Pelanggan")
print(f"{'='*60}")

print("\n" + "="*80)
print("6. UJI T (Uji Signifikansi Koefisien Regresi)")
print("="*80)

# Uji t untuk b1
t_hitung = b1 / Sb1

# t tabel (two-tailed) dengan alpha = 0.05, df = 18
t_tabel = 2.101  # t(0.025, 18) untuk two-tailed test

print(f"\nUji t untuk Koefisien Regresi (b1):")
print(f"t hitung = b1 / Sb1")
print(f"t hitung = {b1:.6f} / {Sb1:.6f}")
print(f"t hitung = {t_hitung:.6f}")

print(f"\n{'='*40}")
print(f"t hitung = {t_hitung:.4f}")
print(f"t tabel (α=0.05, df={df_residual}, two-tailed) = ±{t_tabel:.3f}")
print(f"{'='*40}")

print(f"\nHipotesis:")
print(f"H0: β1 = 0 (Pelayanan tidak berpengaruh signifikan terhadap Kepuasan)")
print(f"H1: β1 ≠ 0 (Pelayanan berpengaruh signifikan terhadap Kepuasan)")

print(f"\nTingkat Signifikansi: α = 0.05 (5%)")
print(f"Derajat Bebas: df = {df_residual}")
print(f"Jenis Uji: Two-tailed (dua arah)")

print(f"\nKriteria Pengujian:")
print(f"Jika |t hitung| > t tabel, maka H0 ditolak")
print(f"Jika |t hitung| ≤ t tabel, maka H0 diterima")

print(f"\n{'='*60}")
print(f"KESIMPULAN UJI T:")
if abs(t_hitung) > t_tabel:
    print(f"|t hitung| = {abs(t_hitung):.4f} > t tabel = {t_tabel:.3f}")
    print(f"H0 DITOLAK, H1 DITERIMA")
    print(f"\nKoefisien regresi SIGNIFIKAN pada tingkat kepercayaan 95%")
    print(f"Variabel Pelayanan berpengaruh SIGNIFIKAN terhadap")
    print(f"Kepuasan Pelanggan")
    print(f"\nSetiap peningkatan 1 poin dalam Pelayanan akan meningkatkan")
    print(f"Kepuasan Pelanggan sebesar {b1:.4f} poin")
else:
    print(f"|t hitung| = {abs(t_hitung):.4f} ≤ t tabel = {t_tabel:.3f}")
    print(f"H0 DITERIMA")
    print(f"\nKoefisien regresi TIDAK SIGNIFIKAN")
    print(f"Variabel Pelayanan tidak berpengaruh signifikan")
    print(f"terhadap Kepuasan Pelanggan")
print(f"{'='*60}")

# Uji t untuk b0 (opsional)
print(f"\n\nUji t untuk Konstanta (b0) - Opsional:")
t_hitung_b0 = b0 / Sb0
print(f"t hitung (b0) = b0 / Sb0 = {b0:.6f} / {Sb0:.6f} = {t_hitung_b0:.4f}")
if abs(t_hitung_b0) > t_tabel:
    print(f"Konstanta signifikan (|{t_hitung_b0:.4f}| > {t_tabel:.3f})")
else:
    print(f"Konstanta tidak signifikan (|{t_hitung_b0:.4f}| ≤ {t_tabel:.3f})")

print("\n" + "="*80)
print("RINGKASAN HASIL ANALISIS")
print("="*80)

print(f"\n1. PERSAMAAN REGRESI:")
print(f"   Ŷ = {b0:.4f} + {b1:.4f}X")
print(f"   Kepuasan Pelanggan = {b0:.4f} + {b1:.4f}(Pelayanan)")

print(f"\n2. KOEFISIEN DETERMINASI:")
print(f"   R² = {R_squared:.4f} atau {R_squared*100:.2f}%")
print(f"   Interpretasi: {R_squared*100:.2f}% variasi kepuasan pelanggan dapat")
print(f"   dijelaskan oleh pelayanan")

print(f"\n3. STANDARD ERROR OF ESTIMATE:")
print(f"   Se = {Se:.4f}")
print(f"   Interpretasi: Rata-rata kesalahan prediksi = {Se:.4f} poin")

print(f"\n4. STANDARD ERROR KOEFISIEN REGRESI:")
print(f"   Sb1 = {Sb1:.6f}")
print(f"   Sb0 = {Sb0:.6f}")

print(f"\n5. UJI F:")
print(f"   F hitung = {F_hitung:.4f}")
print(f"   F tabel = {F_tabel:.2f}")
print(f"   Kesimpulan: Model regresi {'SIGNIFIKAN' if F_hitung > F_tabel else 'TIDAK SIGNIFIKAN'}")

print(f"\n6. UJI T:")
print(f"   t hitung = {t_hitung:.4f}")
print(f"   t tabel = ±{t_tabel:.3f}")
print(f"   Kesimpulan: Pelayanan berpengaruh {'SIGNIFIKAN' if abs(t_hitung) > t_tabel else 'TIDAK SIGNIFIKAN'}")
print(f"   terhadap Kepuasan Pelanggan")

print("\n" + "="*80)
print("KESIMPULAN AKHIR")
print("="*80)

print(f"""
Berdasarkan hasil analisis regresi linear sederhana dengan menggunakan
data dari {n} responden, dapat disimpulkan bahwa:

1. Persamaan regresi yang terbentuk adalah:
   Ŷ = {b0:.4f} + {b1:.4f}X

2. Nilai R² = {R_squared*100:.2f}% menunjukkan bahwa model dapat menjelaskan
   {R_squared*100:.2f}% variasi kepuasan pelanggan.

3. Hasil Uji F menunjukkan bahwa model regresi signifikan (F hitung =
   {F_hitung:.4f} > F tabel = {F_tabel:.2f}), yang berarti variabel
   pelayanan secara bersama-sama berpengaruh signifikan terhadap
   kepuasan pelanggan.

4. Hasil Uji t menunjukkan bahwa koefisien regresi signifikan
   (|t hitung| = {abs(t_hitung):.4f} {'>' if abs(t_hitung) > t_tabel else '≤'} t tabel = {t_tabel:.3f}), yang berarti
   variabel pelayanan berpengaruh {'signifikan' if abs(t_hitung) > t_tabel else 'tidak signifikan'} terhadap kepuasan
   pelanggan.

5. Dengan demikian, dapat disimpulkan bahwa terdapat pengaruh yang
   {'positif dan signifikan' if b1 > 0 and abs(t_hitung) > t_tabel else 'tidak signifikan'} dari pelayanan terhadap kepuasan pelanggan
   pada tingkat kepercayaan 95% (α = 0.05).
""")

print("="*80)

# Membuat tabel prediksi
print("\n" + "="*80)
print("TABEL PREDIKSI DAN RESIDUAL")
print("="*80)

residuals = [kepuasan[i] - Y_pred[i] for i in range(n)]
residuals_squared = [r**2 for r in residuals]

print(f"\n{'No':<5} {'X':<8} {'Y':<8} {'Ŷ':<10} {'e=Y-Ŷ':<10} {'e²':<10}")
print("-" * 55)
for i in range(n):
    print(f"{i+1:<5} {pelayanan[i]:<8} {kepuasan[i]:<8} {Y_pred[i]:<10.4f} {residuals[i]:<10.4f} {residuals_squared[i]:<10.6f}")

print("-" * 55)
print(f"{'TOTAL':<29} {sum(residuals_squared):.6f}")
print(f"\nΣe² = {sum(residuals_squared):.6f} (SS Residual)")

print("\n" + "="*80)
print("ANALISIS SELESAI")
print("="*80)
