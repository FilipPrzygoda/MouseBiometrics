"""
Prosty test weryfikujący strukturę modułów
"""

import sys
sys.path.insert(0, '/home/kali/PW/analiza_danych/project')

from normalizacja import BiometricNormalizer
import pandas as pd

print("=" * 60)
print("TEST NORMALIZACJI")
print("=" * 60)

# Test 1: Inicjalizacja normalizatora
print("\n[TEST 1] Inicjalizacja BiometricNormalizer...")
normalizer = BiometricNormalizer("test_user")
print("✓ BiometricNormalizer zainicjalizowany")

# Test 2: Ekstrakcja cech - dane treningu (z is_target_user)
print("\n[TEST 2] Ekstrakcja cech do treningu (z is_target_user)...")
sample_session = {
    'events': [
        {'timestamp': 100, 'x': 10, 'y': 20, 'type': 'mousemove', 'score': 1, 'diamondLeft': 50, 'diamondTop': 50},
        {'timestamp': 110, 'x': 15, 'y': 25, 'type': 'mousemove', 'score': 1, 'diamondLeft': 50, 'diamondTop': 50},
        {'timestamp': 120, 'x': 75, 'y': 75, 'type': 'diamond_click', 'score': 1, 'diamondLeft': 50, 'diamondTop': 50},
    ]
}

features_train = normalizer.extract_features(sample_session, is_target_user=True, session_id='test_1')
print(f"✓ Ekstrakcja treningu: {len(features_train)} rekordów")
if features_train:
    print(f"  Kolumny: {list(features_train[0].keys())}")
    print(f"  Czy ma 'is_target_user'? {'is_target_user' in features_train[0]}")
    print(f"  Czy ma 'session_id'? {'session_id' in features_train[0]}")

# Test 3: Ekstrakcja cech - dane predykcji (bez is_target_user)
print("\n[TEST 3] Ekstrakcja cech do predykcji (bez is_target_user)...")
df_pred = normalizer.normalize_features_for_prediction(sample_session)
print(f"✓ Ekstrakcja predykcji: DataFrame z {len(df_pred)} wierszy")
print(f"  Kolumny: {list(df_pred.columns)}")
print(f"  Czy ma 'is_target_user'? {'is_target_user' in df_pred.columns}")
print(f"  Czy ma 'session_id'? {'session_id' in df_pred.columns}")

# Test 4: Walidacja kolumn
print("\n[TEST 4] Walidacja kolumn...")
expected_cols = ['duration_ms', 'ideal_distance', 'actual_distance', 'path_efficiency', 'avg_speed', 'points_recorded', 'click_error_distance']
actual_cols = list(df_pred.columns)
missing = set(expected_cols) - set(actual_cols)
extra = set(actual_cols) - set(expected_cols)

if not missing and not extra:
    print("✓ Kolumny się zgadzają!")
else:
    if missing:
        print(f"✗ Brakujące kolumny: {missing}")
    if extra:
        print(f"✗ Dodatkowe kolumny: {extra}")

print("\n" + "=" * 60)
print("TESTY ZAKOŃCZONE")
print("=" * 60)
