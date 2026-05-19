# Architektura Modułów Biometrycznych

## Struktura

### 1. **normalizacja.py** - BiometricNormalizer
- **Cel**: Ekstrakcja cech z surowych danych zdarzeń myszy
- **Główne metody**:
  - `extract_features(session_data, is_target_user=None, session_id=None)` - Ekstrakcja cech z sesji
  - `prepare_dataset_for_training(user_sessions, other_sessions)` - Przygotowanie datasetu do treningu
  - `normalize_features_for_prediction(session_data)` - Normalizacja danych do predykcji

- **Wyekstrahowane cechy**:
  - `duration_ms` - Czas trwania ruchu
  - `ideal_distance` - Dystans teoretyczny (od startu do środka diamentu)
  - `actual_distance` - Faktyczna długość ścieżki
  - `path_efficiency` - Wydajność ścieżki (ideal/actual)
  - `avg_speed` - Średnia prędkość
  - `points_recorded` - Liczba zarekordowanych punktów
  - `click_error_distance` - Błąd precyzji kliknięcia (dystans od środka diamentu)
  - `is_target_user` - **Tylko do treningu** (1=autentyczny, 0=anomalia)

### 2. **trening.py** - BiometricTrainer
- **Cel**: Trenowanie modelu Random Forest i zapis do pliku
- **Inicjalizacja**: `BiometricTrainer(username, db_uri='mongodb://localhost:27017/', db_name='biometria_db')`
- **Główne metody**:
  - `train()` - Pobiera dane z bazy, trenuje model, zapisuje
  - `save_model()` - Ręczny zapis modelu
  - `load_model()` - Ładowanie istniejącego modelu
  
- **Proces**:
  1. Pobiera sesje użytkownika i inne sesje z bazy (50/50 balans)
  2. Ekstrakcja cech za pomocą `BiometricNormalizer`
  3. Podział na trening/test (70/30 po sesjach)
  4. Skalowanie cech (StandardScaler)
  5. Trening Random Forest
  6. Zapis modelu i skalera do `models/`

- **Przechowywanie**:
  - Model: `models/model_{username}.pkl`
  - Skaler: `models/scaler_{username}.pkl`

### 3. **decyzja.py** - BiometricDecision
- **Cel**: Podejmowanie decyzji autentykacji na bazie przesłanych danych
- **Inicjalizacja**: `BiometricDecision(username)`
- **Główne metody**:
  - `predict(events)` - Predykcja z aggregacją (majority voting)
    - Zwraca: `{'is_correct_user': bool, 'confidence': float}`
  - `predict_with_details(events)` - Predykcja z dodatkowymi szczegółami
  
- **Agregacja**: Używa majority voting >= 0.5 na wszystkich ruchach w sesji

## Integración z app.py

### Endpointy

- **POST `/api/biometrics`** - Zapis surowych danych do bazy (bez zmian)
- **POST `/api/recognize`** - Predykcja autentykacji
  - Używa: `BiometricDecision`
  - Zwraca: `is_correct`, `confidence`
  
- **POST `/api/train`** - Trening modelu
  - Używa: `BiometricTrainer`
  - Automatycznie pobiera dane z bazy
  
- **GET `/api/model-status`** - Sprawdzenie czy model jest wytrenowany

## Różnice vs `test.py`

1. **Moduły** - Kod podzielony na trzy moduły z czytelną separacją odpowiedzialności
2. **Random Forest** - Zamiast Isolation Forest
3. **Agregacja sesji** - Majority voting na całej sesji (>= 0.5)
4. **Bez is_target_user w predykcji** - Dane z przeglądarki nie zawierają tej informacji
5. **Integracja z Flask** - Bezpośrednie użycie w endpointach

## Workflow

### Rejestracja/Trening
```
1. Użytkownik wykonuje serię ruchów i dane zapisywane do bazy
2. Admin/użytkownik wywołuje POST /api/train
3. BiometricTrainer pobiera sesje z bazy
4. Ekstrakcja cech via BiometricNormalizer
5. Trening Random Forest
6. Model zapisywany do models/model_{username}.pkl
```

### Autentykacja
```
1. Użytkownik wykonuje ruchy na stronie
2. POST /api/recognize z danymi zdarzeń
3. BiometricDecision ekstrakcje cechy (bez is_target_user)
4. Skalowanie
5. Predykcja dla każdego ruchu
6. Aggregation via majority voting
7. Zwrot decyzji (is_correct_user, confidence)
```
