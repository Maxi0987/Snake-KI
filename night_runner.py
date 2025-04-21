import subprocess
import os
import platform
import time

def run_with_retry(script_name, max_retries=10, wait_between=15):
    retries = 0
    while retries < max_retries:
        print(f"▶️ Starte {script_name} (Versuch {retries + 1}/{max_retries})...")
        process = subprocess.run(["python", script_name])
        
        # ✅ Erfolg wenn `train_done.flag` existiert
        if os.path.exists("train_done.flag"):
            print(f"✅ {script_name} abgeschlossen.")
            return True

        print(f"⚠️ {script_name} wurde unerwartet beendet. Neuer Versuch in {wait_between} Sekunden...")
        retries += 1
        time.sleep(wait_between)

    print("❌ Maximale Anzahl an Versuchen erreicht – Abbruch.")
    return False

print("🌙 Starte Nachttraining...")

# Schritt 1: Training mit Wiederholungen
if run_with_retry("train_later.py"):
    # Schritt 2: Analyse starten
    print("\n📊 Starte find_average_score.py...")
    subprocess.run(["python", "find_average_score.py"])
    print("🏁 Auswertung abgeschlossen.")

    # Schritt 3: Shutdown
    print("🛑 Fahre den PC in 30 Sekunden herunter... Speichern nicht vergessen!")
    time.sleep(30)

    system_platform = platform.system()

    if system_platform == "Windows":
        os.system("shutdown /s /t 0")
    elif system_platform in ["Linux", "Darwin"]:
        os.system("sudo shutdown -h now")
    else:
        print("❌ Unbekanntes Betriebssystem – Shutdown nicht unterstützt.")
else:
    print("🚫 Nachttraining fehlgeschlagen – kein Shutdown.")
