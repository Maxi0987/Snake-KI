import subprocess
import os
import platform
import time

print("🌙 Starte Nachttraining...")

# Schritt 1: Trainieren
print("▶️ Starte train_later.py...")
subprocess.run(["python", "train_later.py"])
print("✅ Training abgeschlossen.\n")

# Schritt 2: Auswertung
print("📊 Starte find_average_score.py...")
subprocess.run(["python", "find_average_score.py"])
print("🏁 Auswertung abgeschlossen.")

# Schritt 3: Shutdown
print("🛑 Fahre den PC in 30 Sekunden herunter... Speichern nicht vergessen!")
time.sleep(30)

system_platform = platform.system()

if system_platform == "Windows":
    os.system("shutdown /s /t 0")
elif system_platform == "Linux" or system_platform == "Darwin":
    os.system("sudo shutdown -h now")
else:
    print("❌ Unbekanntes Betriebssystem – Shutdown nicht unterstützt.")
