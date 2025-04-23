import os
import tensorflow as tf

print("\n🔍 Starte XLA/GPU-Test\n")

# Aktiviert XLA explizit
tf.config.optimizer.set_jit(True)

# Zeigt erkannte Geräte an
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU erkannt: {gpus[0].name}")
else:
    print("❌ Keine GPU erkannt – Training läuft nur auf CPU")

# Funktion, die XLA verwenden soll
@tf.function(jit_compile=True)
def xla_matmul(a, b):
    return tf.matmul(a, b)

try:
    # Dummy-Daten für einfachen MatMul
    a = tf.random.normal((1024, 1024))
    b = tf.random.normal((1024, 1024))
    
    print("🚀 Führe XLA-Matrixmultiplikation durch...")
    result = xla_matmul(a, b)
    
    print("✅ XLA-Kompatible Operation erfolgreich durchgeführt.")
    print("📐 Ergebnisform:", result.shape)

except tf.errors.InternalError as e:
    print("❌ XLA-Fehler:", e.message)
    print("💡 Mögliche Ursache: Kein Speicherplatz? Schreibrechte auf TEMP-Ordner?")

except Exception as e:
    print("❌ Allgemeiner Fehler:", e)

print("\n🧪 Test abgeschlossen.\n")
