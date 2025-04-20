import tensorflow as tf

print("✅ TensorFlow Version:", tf.__version__)
print("🧠 Geräte:")
print(tf.config.list_physical_devices())
print("📦 GPU verfügbar:", tf.config.list_physical_devices('GPU'))

# Zusätzliche Infos
from tensorflow.python.client import device_lib
print("\n🔍 Vollständige Geräteliste:")
for d in device_lib.list_local_devices():
    print(f"{d.name} | {d.device_type}")
