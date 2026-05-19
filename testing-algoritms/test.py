import random
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. FUNKCJE POBIERANIA I PRZYGOTOWANIA DANYCH
# ==========================================

def pobierz_surowe_dane(uri, db_name, collection_name):
    """Pobiera wszystkie rekordy z bazy MongoDB."""
    client = MongoClient(uri)
    db = client[db_name]
    return list(db[collection_name].find({}, {'_id': 0}))

def przygotuj_zbalansowane_sesje(username, raw_data):
    """Dzieli sesje na użytkownika docelowego i resztę, balansując zbiór 50/50."""
    user_sessions = [r for r in raw_data if r.get('username') == username]
    other_sessions = [r for r in raw_data if r.get('username') != username]
    
    if len(user_sessions) < len(other_sessions):
        random.shuffle(other_sessions)
        other_sessions = other_sessions[:len(user_sessions)]
        
    return user_sessions, other_sessions

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def extract_features(session_data, is_target_user, session_id):
    events = session_data.get('events', [])
    if not events: return []
        
    df = pd.DataFrame(events)
    if 'type' not in df.columns or 'score' not in df.columns: return []
    
    # Nie odfiltrowujemy kliknięć, żeby mieć dostęp do ostatniej koordynaty (X, Y)
    if df.empty: return []
        
    features_list = []
    
    # ZMIENNA POMOCNICZA: Rozmiar diamentu w pikselach (np. 50x50).
    # Dostosuj tę wartość do rzeczywistych wymiarów z Twojego front-endu!
    DIAMOND_WIDTH = 50 
    DIAMOND_HEIGHT = 50

    for score, trajectory in df.groupby('score'):
        trajectory = trajectory.sort_values('timestamp')
        
        # Odrzucamy puste trajektorie
        if trajectory['x'].isnull().all() or trajectory['y'].isnull().all():
            continue
            
        # 1. Czas trwania
        t_start, t_end = trajectory['timestamp'].iloc[0], trajectory['timestamp'].iloc[-1]
        duration_ms = max(t_end - t_start, 1) 
        
        # 2. Wyliczenie ŚRODKA diamentu
        # Zakładamy, że diamondLeft/Top to lewy górny róg elementu.
        diamond_left = trajectory['diamondLeft'].iloc[0]
        diamond_top = trajectory['diamondTop'].iloc[0]
        
        center_x = diamond_left + (DIAMOND_WIDTH / 2)
        center_y = diamond_top + (DIAMOND_HEIGHT / 2)
        
        # 3. Odległości (idealna i pokonana)
        start_x, start_y = trajectory['x'].iloc[0], trajectory['y'].iloc[0]
        ideal_distance = calculate_distance(start_x, start_y, center_x, center_y)
        
        x_coords, y_coords = trajectory['x'].values, trajectory['y'].values
        actual_distance = np.sum([calculate_distance(x_coords[i-1], y_coords[i-1], x_coords[i], y_coords[i]) 
                                  for i in range(1, len(x_coords))])
        
        # 4. Wskaźniki dynamiki
        efficiency = ideal_distance / actual_distance if actual_distance > 0 else 0
        avg_speed = actual_distance / duration_ms
        
        # 5. NOWA CECHA: Precyzja kliknięcia (błąd odległości od środka)
        # Bierzemy ostatnie zarejestrowane koordynaty (czyli moment kliknięcia)
        last_x, last_y = trajectory['x'].iloc[-1], trajectory['y'].iloc[-1]
        click_error_distance = calculate_distance(last_x, last_y, center_x, center_y)
        
        features_list.append({
            'session_id': session_id,
            'duration_ms': duration_ms,
            'ideal_distance': ideal_distance,
            'actual_distance': actual_distance,
            'path_efficiency': efficiency,
            'avg_speed': avg_speed,
            'points_recorded': len(trajectory),
            'click_error_distance': click_error_distance, # <--- Precyzja kliknięcia
            'is_target_user': 1 if is_target_user else 0
        })
        
    return features_list

def prepare_dataset(user_sessions, other_sessions):
    """Agreguje cechy ze wszystkich sesji w jedną ramkę danych."""
    all_features = []
    for i, session in enumerate(user_sessions):
        all_features.extend(extract_features(session, True, f"user_{i}"))
    for i, session in enumerate(other_sessions):
        all_features.extend(extract_features(session, False, f"other_{i}"))
        
    df = pd.DataFrame(all_features)
    return df.sample(frac=1).reset_index(drop=True) if not df.empty else df

# ==========================================
# 2. FUNKCJE EWALUACYJNE I WIZUALIZACYJNE
# ==========================================

def calculate_metrics(y_true, y_pred):
    """Liczy macierz pomyłek oraz metryki FAR, FRR, ACC."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return cm, far, frr, acc, (tn, fp, fn, tp)

def evaluate_whole_session(df_test, y_pred_event):
    """Agreguje predykcje z eventów do oceny całej sesji (majority voting >= 50%)."""
    df_res = df_test[['session_id', 'is_target_user']].copy()
    df_res['pred_event'] = y_pred_event
    
    session_grp = df_res.groupby('session_id').agg(
        actual_label=('is_target_user', 'first'),
        pred_mean=('pred_event', 'mean')
    )
    session_grp['session_pred'] = (session_grp['pred_mean'] >= 0.5).astype(int)
    
    return calculate_metrics(session_grp['actual_label'], session_grp['session_pred'])

def plot_and_save_cm(cm_event, cm_session, filepath):
    """Rysuje i zapisuje podwójny wykres macierzy pomyłek."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm_event, annot=True, fmt='g', cmap='Blues', ax=axes[0], 
                xticklabels=['Anomalia (0)', 'Autentyczny (1)'], yticklabels=['Anomalia (0)', 'Autentyczny (1)'])
    axes[0].set_title('Macierz Pomyłek - Pojedynczy Event')
    axes[0].set_ylabel('Rzeczywistość')

    sns.heatmap(cm_session, annot=True, fmt='g', cmap='Greens', ax=axes[1], 
                xticklabels=['Anomalia (0)', 'Autentyczny (1)'], yticklabels=['Anomalia (0)', 'Autentyczny (1)'])
    axes[1].set_title('Macierz Pomyłek - Cała Sesja')
    axes[1].set_ylabel('Rzeczywistość')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    plt.savefig(filepath)
    plt.close()

# ==========================================
# 3. GŁÓWNY POTOK (MAIN)
# ==========================================

if __name__ == "__main__":
    # --- KONFIGURACJA ---
    USERNAME = "filipp"
    DB_URI = "mongodb://localhost:27017/"
    DB_NAME = "biometria_db"
    COLLECTION = "sesje_uzytkownikow"
    CM_FILEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_matrices.png")

    # --- KROK 1: POBIERANIE DANYCH ---
    print(f"Pobieranie danych z bazy dla użytkownika: {USERNAME}...")
    surowe_dane = pobierz_surowe_dane(DB_URI, DB_NAME, COLLECTION)
    user_sessions, other_sessions = przygotuj_zbalansowane_sesje(USERNAME, surowe_dane)
    print(f"-> Znaleziono {len(user_sessions)} sesji autentycznych i {len(other_sessions)} sesji anomalii.")

    # --- KROK 2: EKSTRAKCJA CECH ---
    df_model = prepare_dataset(user_sessions, other_sessions)
    if df_model.empty:
        print("\nBłąd: Brak wystarczających danych do zbudowania zbioru treningowego.")
        exit()

    # --- KROK 3: PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY (PO SESJACH) ---
    unique_sessions = df_model['session_id'].unique()
    train_sessions, test_sessions = train_test_split(unique_sessions, test_size=0.3, random_state=2)

    df_train = df_model[df_model['session_id'].isin(train_sessions)]
    df_test = df_model[df_model['session_id'].isin(test_sessions)]

    print("\n[ PODZIAŁ ZBIORU ]")
    print(f"Sesje (Trening / Test): {len(train_sessions)} / {len(test_sessions)}")
    print(f"Pojedyncze ruchy (Trening / Test): {len(df_train)} / {len(df_test)}")

    # Izolowanie cech i etykiet
    cols_to_drop = ['is_target_user', 'session_id']
    X_train, y_train = df_train.drop(cols_to_drop, axis=1), df_train['is_target_user']
    X_test, y_test = df_test.drop(cols_to_drop, axis=1), df_test['is_target_user']

    # --- KROK 4: TRENING MODELU ---
    print("\nTrenowanie modelu Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # --- KROK 5: PREDYKCJA ---
    y_pred_event = rf_model.predict(X_test)

    # --- KROK 6: EWALUACJA ---
    cm_event, far_e, frr_e, acc_e, (tn_e, fp_e, fn_e, tp_e) = calculate_metrics(y_test, y_pred_event)
    cm_session, far_s, frr_s, acc_s, (tn_s, fp_s, fn_s, tp_s) = evaluate_whole_session(df_test, y_pred_event)

    print("\n[ WYNIKI - POJEDYNCZE RUCHY (EVENTS) ]")
    print(f"Dokładność: {acc_e:.4f} | FAR: {far_e:.4f} | FRR: {frr_e:.4f}")
    print(f"TN: {tn_e} | FP: {fp_e} | FN: {fn_e} | TP: {tp_e}")

    print("\n[ WYNIKI - CAŁE SESJE (MAJORITY VOTING >= 50%) ]")
    print(f"Dokładność: {acc_s:.4f} | FAR: {far_s:.4f} | FRR: {frr_s:.4f}")
    print(f"TN: {tn_s} | FP: {fp_s} | FN: {fn_s} | TP: {tp_s}")

    # --- KROK 7: WIZUALIZACJA ---
    plot_and_save_cm(cm_event, cm_session, CM_FILEPATH)
    print(f"\nZapisano wykresy macierzy pomyłek do: {CM_FILEPATH}")