"""
Moduł normalizacji - ekstrakcja cech z surowych danych biometrycznych.
Konwertuje zdarzenia myszy na wektor cech do trenowania i predykcji.
"""

import math
import pandas as pd
import numpy as np


def calculate_distance(x1, y1, x2, y2):
    """Oblicza odległość euklidesową między dwoma punktami."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class BiometricNormalizer:
    """
    Klasa odpowiadająca za normalizację i ekstrakcję cech z surowych danych.
    """
    
    # Rozmiar diamentu w pikselach (dostosuj do rzeczywistych wymiarów z front-endu)
    DIAMOND_WIDTH = 50
    DIAMOND_HEIGHT = 50
    
    def __init__(self, username):
        self.username = username
    
    def extract_features(self, session_data, is_target_user=None, session_id=None, include_session_id=True):
        """
        Ekstrakcja cech z pojedynczej sesji.
        
        Args:
            session_data: Dict zawierający 'events' - lista zdarzeń myszy
            is_target_user: Bool (dla treningu) lub None (dla predykcji)
            session_id: String identyfikator sesji
            include_session_id: Bool - czy dodawać session_id do output (False dla predykcji)
            
        Returns:
            Lista dict'ów z wyekstrahowanymi cechami, lub pusta lista
        """
        events = session_data.get('events', [])
        if not events:
            return []
        
        df = pd.DataFrame(events)
        if 'type' not in df.columns or 'score' not in df.columns:
            return []
        
        if df.empty:
            return []
        
        features_list = []
        
        # Grupowanie po score (każdy ruch do diamentu)
        for score, trajectory in df.groupby('score'):
            trajectory = trajectory.sort_values('timestamp')
            
            # Odrzucamy puste trajektorie
            if trajectory['x'].isnull().all() or trajectory['y'].isnull().all():
                continue
            
            # 1. Czas trwania
            t_start = trajectory['timestamp'].iloc[0]
            t_end = trajectory['timestamp'].iloc[-1]
            duration_ms = max(t_end - t_start, 1)
            
            # 2. Wyliczenie ŚRODKA diamentu
            diamond_left = trajectory['diamondLeft'].iloc[0]
            diamond_top = trajectory['diamondTop'].iloc[0]
            
            center_x = diamond_left + (self.DIAMOND_WIDTH / 2)
            center_y = diamond_top + (self.DIAMOND_HEIGHT / 2)
            
            # 3. Odległości (idealna i pokonana)
            start_x = trajectory['x'].iloc[0]
            start_y = trajectory['y'].iloc[0]
            ideal_distance = calculate_distance(start_x, start_y, center_x, center_y)
            
            x_coords = trajectory['x'].values
            y_coords = trajectory['y'].values
            actual_distance = np.sum([
                calculate_distance(x_coords[i-1], y_coords[i-1], x_coords[i], y_coords[i])
                for i in range(1, len(x_coords))
            ])
            
            # 4. Wskaźniki dynamiki
            efficiency = ideal_distance / actual_distance if actual_distance > 0 else 0
            avg_speed = actual_distance / duration_ms if duration_ms > 0 else 0
            
            # 5. Precyzja kliknięcia (błąd odległości od środka)
            last_x = trajectory['x'].iloc[-1]
            last_y = trajectory['y'].iloc[-1]
            click_error_distance = calculate_distance(last_x, last_y, center_x, center_y)
            
            # Zbudowanie feature dictionary
            feature_dict = {
                'duration_ms': float(duration_ms),
                'ideal_distance': float(ideal_distance),
                'actual_distance': float(actual_distance),
                'path_efficiency': float(efficiency),
                'avg_speed': float(avg_speed),
                'points_recorded': len(trajectory),
                'click_error_distance': float(click_error_distance),
            }
            
            # Dodaj session_id jeśli potrzebny (do treningu)
            if include_session_id:
                feature_dict['session_id'] = session_id if session_id else 'unknown'
            
            # Dodaj is_target_user tylko jeśli podany (dla treningu)
            if is_target_user is not None:
                feature_dict['is_target_user'] = 1 if is_target_user else 0
            
            features_list.append(feature_dict)
        
        return features_list
    
    def prepare_dataset_for_training(self, user_sessions, other_sessions):
        """
        Przygotowuje dataset do trenowania modelu.
        
        Args:
            user_sessions: Lista sesji użytkownika docelowego
            other_sessions: Lista sesji innych użytkowników (anomalii)
            
        Returns:
            DataFrame z wszystkimi cechami
        """
        all_features = []
        
        for i, session in enumerate(user_sessions):
            features = self.extract_features(session, is_target_user=True, session_id=f"user_{i}")
            all_features.extend(features)
        
        for i, session in enumerate(other_sessions):
            features = self.extract_features(session, is_target_user=False, session_id=f"other_{i}")
            all_features.extend(features)
        
        if not all_features:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_features)
        # Zmieszaj i resetuj indeks
        df = df.sample(frac=1).reset_index(drop=True) if not df.empty else df
        
        return df
    
    def normalize_features_for_prediction(self, session_data):
        """
        Normalizuje features dla predykcji (bez is_target_user i session_id).
        
        Args:
            session_data: Dict zawierający 'events'
            
        Returns:
            DataFrame z cechami do predykcji (bez is_target_user i session_id)
        """
        features = self.extract_features(session_data, is_target_user=None, include_session_id=False)
        
        if not features:
            return pd.DataFrame()
        
        df = pd.DataFrame(features)
        return df
