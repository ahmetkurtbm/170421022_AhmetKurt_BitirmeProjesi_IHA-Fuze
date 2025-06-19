import pandas as pd
import matplotlib.pyplot as plt

# Belirli waypoint'ler (kırmızı olacak)
waypoints_lat = [-5.409957, -5.364957, -5.319957]
waypoints_lon = [-146.100827, -146.055827, -146.010827]

# Veri yükleme ve temizleme (güncellenmiş applymap yerine map kullanımı)
df = pd.read_excel("flight_logs/tb2_flight_data_20250617_193455.xlsx")
df = df.map(lambda x: float(str(x).replace(",", ".")) if isinstance(x, str) else x)

# Metin verisi
metrikler = [
    "Toplam Uçuş Süresi (s)",
    "Maksimum İrtifa (m)",
    "Ortalama İrtifa (m)",
    "Minimum İrtifa (m)",
    "Maksimum Hız (m/s)",
    "Ortalama Hız (m/s)",
    "Maksimum Roll Açısı (°)",
    "Maksimum Pitch Açısı (°)",
    "Başlangıç-Hedef Mesafe (m)",
    "Son-Hedef Mesafe (m)",
    "Waypoint Sayısı"
]

degerler = [
    "202.73",
    "1410.30",
    "1028.27",
    "852.67",
    "64.46",
    "43.23",
    "74.57",
    "26.94",
    "3198.01",
    "NaN",
    "3"
]

# Renkli tablo oluştur
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')  # Eksenleri kapat

# Tabloyu oluştur
table = ax.table(
    cellText=list(zip(metrikler, degerler)),
    colLabels=["Metrik", "Değer"],
    cellLoc='center',
    loc='center',
    colColours=["#4a5568", "#4a5568"],
)

# Stil ayarları
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.4)

# Başlık
plt.title("🚀 Uçuş Performans Özeti", fontsize=14, fontweight="bold", pad=20)

plt.tight_layout()
plt.show()


# Önemli istatistikleri hesapla
flight_stats = {
    "Toplam Uçuş Süresi (s)": f"{df['time'].max() - df['time'].min():.2f}",
    "Maksimum İrtifa (m)": f"{df['altitude'].max():.2f}",
    "Ortalama İrtifa (m)": f"{df['altitude'].mean():.2f}",
    "Minimum İrtifa (m)": f"{df['altitude'].min():.2f}",
    "Maksimum Hız (m/s)": f"{df['velocity'].max():.2f}",
    "Ortalama Hız (m/s)": f"{df['velocity'].mean():.2f}",
    "Maksimum Roll Açısı (°)": f"{df['roll'].max():.2f}",
    "Maksimum Pitch Açısı (°)": f"{df['pitch'].max():.2f}",
    "Başlangıç-Hedef Mesafe (m)": f"{df['distance_to_target'].iloc[0]:.2f}",
    "Son-Hedef Mesafe (m)": f"{df['distance_to_target'].iloc[-1]:.2f}",
    "Waypoint Sayısı": len(waypoints_lat)
}

# DataFrame oluştur ve yazdır
stats_df = pd.DataFrame(list(flight_stats.items()), columns=['Metrik', 'Değer'])

# Görsel olarak düzenli çıktı
print("\n" + "="*60)
print(" UÇUŞ PERFORMANS ÖZET TABLOSU ".center(60, '='))
print("="*60)
print(stats_df.to_string(index=False))
print("="*60)

# Excel'e kaydet
stats_df.to_excel("flight_summary.xlsx", index=False)
print("\nÖzet tablo 'flight_summary.xlsx' olarak kaydedildi.")


# Grafik 1: Zaman - İrtifa
plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["altitude"], color="blue", label="Altitude")
plt.title("Zaman - İrtifa")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Grafik 2: Zaman - Velocity
plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["velocity"], color="green", label="Velocity")
plt.title("Zaman - Hız")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Grafik 3: Zaman - Roll, Pitch, Yaw
plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["roll"], label="Roll")
plt.plot(df["time"], df["pitch"], label="Pitch")
plt.plot(df["time"], df["yaw"], label="Yaw")
plt.title("Zaman - Roll / Pitch / Yaw")
plt.xlabel("Time (s)")
plt.ylabel("Açı (°)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Grafik 4: Konum (latitude vs. longitude)
plt.figure(figsize=(8, 8))
# 1. Rota çizgisi - gri
plt.plot(df["longitude"], df["latitude"], color='gray', linestyle='-', linewidth=1, label="Uçuş Rotası", zorder=1)

# 2. Başlangıç noktası - mavi
plt.scatter(df["longitude"].iloc[0], df["latitude"].iloc[0], color='blue', s=80, label="Başlangıç Noktası", zorder=3)

# 3. Waypoint noktaları - kırmızı
plt.scatter(waypoints_lon, waypoints_lat, color='red', s=60, label="Waypoint Noktaları", zorder=4)

# Opsiyonel: Nokta numaralarını yaz (1, 2, 3)
for i in range(len(waypoints_lat)):
    plt.text(waypoints_lon[i], waypoints_lat[i] + 0.002, f"W{i+1}", color="red", fontsize=10, ha='center')
# Başlangıç noktası - mavi
plt.scatter(df["longitude"].iloc[0], df["latitude"].iloc[0], color='blue', s=80, label="Başlangıç Noktası", zorder=5)
# Bitiş noktası - yeşil
plt.scatter(df["longitude"].iloc[-1], df["latitude"].iloc[-1], color='green', s=80, label="Bitiş Noktası", zorder=5)

plt.title("Harita Üzerinde Uçuş Rotası")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Grafik 5: Zaman - Hedefe Uzaklık
plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["distance_to_target"], color="red", label="Distance to Target")
plt.title("Zaman - Hedefe Olan Uzaklık")
plt.xlabel("Time (s)")
plt.ylabel("Distance to Target (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
