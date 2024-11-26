import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from datetime import datetime, timedelta

class SpotifyMLPipeline:
    def __init__(self, client_id, client_secret, min_tracks=2500):
        """
        Inicializa o pipeline com credenciais do Spotify
        """
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.min_tracks = min_tracks

    def collect_data(self, playlist_ids):
        """
        Coleta dados das playlists do Spotify com cache em arquivo JSON
        Garante um mínimo de 2500 tracks
        """
        cache_file = 'spotify_tracks_cache.json'
        cache_metadata_file = 'spotify_cache_metadata.json'
        
        def is_cache_valid():
            if not os.path.exists(cache_file) or not os.path.exists(cache_metadata_file):
                return False
                
            try:
                with open(cache_metadata_file, 'r') as f:
                    metadata = json.load(f)
                cache_date = datetime.fromisoformat(metadata['last_update'])
                return (datetime.now() - cache_date < timedelta(days=7) and 
                       metadata['total_tracks'] >= self.min_tracks)
            except:
                return False
        
        if is_cache_valid():
            try:
                print("Carregando dados do cache...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    tracks_data = json.load(f)
                self.data = pd.DataFrame(tracks_data)
                print(f"Dados carregados do cache: {len(tracks_data)} músicas")
                return self.data
            except Exception as e:
                print(f"Erro ao carregar cache: {str(e)}")
                print("Coletando dados da API...")
        else:
            print("Cache não encontrado ou expirado. Coletando dados da API...")
        
        tracks_data = []
        tracks_processed = set()  
        
        try:
            while len(tracks_data) < self.min_tracks:
                for playlist_id in playlist_ids:
                    print(f"Processando playlist: {playlist_id}")
                    
                    try:
                        results = self.sp.playlist_tracks(playlist_id)
                        tracks = results['items']
                        
                        while results['next']:
                            time.sleep(1)  
                            results = self.sp.next(results)
                            tracks.extend(results['items'])
                        
                        track_ids = []
                        
                        for item in tracks:
                            track = item['track']
                            if track is None or track['id'] in tracks_processed:
                                continue
                            
                            tracks_processed.add(track['id'])
                            track_ids.append(track['id'])
                            
                            if len(track_ids) == 50:
                                try:
                                    time.sleep(1)  
                                    audio_features_batch = self.sp.audio_features(track_ids)
                                    
                                    for i, features in enumerate(audio_features_batch):
                                        if features:
                                            track_info = [t['track'] for t in tracks if t['track'] and t['track']['id'] == track_ids[i]][0]
                                            track_data = {
                                                'id': track_info['id'],
                                                'name': track_info['name'],
                                                'popularity': track_info['popularity'],
                                                'danceability': features['danceability'],
                                                'energy': features['energy'],
                                                'key': features['key'],
                                                'loudness': features['loudness'],
                                                'mode': features['mode'],
                                                'speechiness': features['speechiness'],
                                                'instrumentalness': features['instrumentalness'],
                                                'liveness': features['liveness'],
                                                'valence': features['valence'],
                                                'tempo': features['tempo'],
                                                'popularity_category': 'high' if track_info['popularity'] >= 70 
                                                                    else 'medium' if track_info['popularity'] >= 40 
                                                                    else 'low'
                                            }
                                            tracks_data.append(track_data)
                                            
                                            if len(tracks_data) % 100 == 0:
                                                print(f"Músicas processadas: {len(tracks_data)}")
                                    
                                    track_ids = []
                                    
                                except Exception as e:
                                    print(f"Erro no lote: {str(e)}")
                                    if "429" in str(e):  
                                        time.sleep(30)
                                    track_ids = []
                                    continue
                        
                     
                        if track_ids:
                            try:
                                time.sleep(1)
                                audio_features_batch = self.sp.audio_features(track_ids)
                                
                                for i, features in enumerate(audio_features_batch):
                                    if features:
                                        track_info = [t['track'] for t in tracks if t['track'] and t['track']['id'] == track_ids[i]][0]
                                        track_data = {
                                            'id': track_info['id'],
                                            'name': track_info['name'],
                                            'popularity': track_info['popularity'],
                                            'danceability': features['danceability'],
                                            'energy': features['energy'],
                                            'key': features['key'],
                                            'loudness': features['loudness'],
                                            'mode': features['mode'],
                                            'speechiness': features['speechiness'],
                                            'instrumentalness': features['instrumentalness'],
                                            'liveness': features['liveness'],
                                            'valence': features['valence'],
                                            'tempo': features['tempo'],
                                            'popularity_category': 'high' if track_info['popularity'] >= 70 
                                                                else 'medium' if track_info['popularity'] >= 40 
                                                                else 'low'
                                        }
                                        tracks_data.append(track_data)
                                
                            except Exception as e:
                                print(f"Erro no último lote: {str(e)}")
                    
                    except Exception as e:
                        print(f"Erro ao processar playlist {playlist_id}: {str(e)}")
                        continue
                
                if len(tracks_data) < self.min_tracks:
                    print(f"Ainda precisamos de {self.min_tracks - len(tracks_data)} músicas. Reiniciando o processo com as mesmas playlists...")
                    tracks_processed.clear() 
            
        except Exception as e:
            print(f"Erro ao coletar dados: {str(e)}")
            if len(tracks_data) < self.min_tracks:
                raise ValueError(f"Não foi possível coletar o mínimo de {self.min_tracks} músicas. Apenas {len(tracks_data)} foram coletadas.")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(tracks_data, f, ensure_ascii=False, indent=2)
            
            with open(cache_metadata_file, 'w') as f:
                json.dump({
                    'last_update': datetime.now().isoformat(),
                    'total_tracks': len(tracks_data),
                    'playlists': playlist_ids
                }, f, indent=2)
                
            print("Dados salvos em cache")
            
        except Exception as e:
            print(f"Erro ao salvar cache: {str(e)}")
        
        self.data = pd.DataFrame(tracks_data)
        print(f"Total de músicas coletadas: {len(tracks_data)}")
        return self.data

    def visualize_data_distribution(self):
        """
        Cria visualizações simplificadas da distribuição dos dados
        """
        if self.data is None:
            raise ValueError("Dados não coletados ainda.")
        
        plt.style.use('dark_background')
        colors = ['#1DB954', '#1ED760', '#169C46']
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Gráfico de Pizza - Distribuição de Popularidade
        plt.subplot(221)
        popularity_counts = self.data['popularity_category'].value_counts()
        plt.pie(popularity_counts, 
                labels=['Alta Popularidade', 'Média Popularidade', 'Baixa Popularidade'],
                autopct='%1.1f%%', 
                colors=colors,
                explode=(0.1, 0, 0))
        plt.title('Distribuição de Músicas por Popularidade', fontsize=14, pad=20)
        
        plt.figtext(0.24, 0.85,
                   'Como as músicas se dividem:\n' +
                   '- Alta: popularidade >= 70\n' +
                   '- Média: popularidade 40-69\n' +
                   '- Baixa: popularidade < 40',
                   bbox=dict(facecolor='white', alpha=0.8),
                   color='black',
                   fontsize=10)
        
        # 2. Características Principais - Gráfico de Barras Horizontais
        plt.subplot(222)
        features = {
            'Energia': self.data['energy'].mean(),
            'Dançabilidade': self.data['danceability'].mean(),
            'Positividade': self.data['valence'].mean(),
        }
        
        bars = plt.barh(list(features.keys()), list(features.values()), color='#1DB954')
        plt.title('Características Principais das Músicas', fontsize=14, pad=20)
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2%}', ha='left', va='center')
        
        # 3. Perfil Musical Simplificado por Categoria
        plt.subplot(223)
        categories = ['Baixa', 'Média', 'Alta']
        energies = [self.data[self.data['popularity_category'] == cat]['energy'].mean() 
                   for cat in ['low', 'medium', 'high']]
        
        plt.bar(categories, energies, color='#1DB954')
        plt.title('Nível de Energia por Popularidade', fontsize=14, pad=20)
        plt.xlabel('Categoria de Popularidade')
        plt.ylabel('Nível Médio de Energia')
        
        for i, v in enumerate(energies):
            plt.text(i, v, f'{v:.2%}', ha='center', va='bottom')
        
        # 4. BPM por Categoria
        plt.subplot(224)
        avg_tempo = self.data.groupby('popularity_category')['tempo'].mean()
        plt.bar(['Baixa', 'Média', 'Alta'], 
                [avg_tempo['low'], avg_tempo['medium'], avg_tempo['high']],
                color='#1DB954')
        plt.title('BPM Médio por Popularidade', fontsize=14, pad=20)
        plt.xlabel('Categoria de Popularidade')
        plt.ylabel('Batidas por Minuto (BPM)')
        
        for i, v in enumerate([avg_tempo['low'], avg_tempo['medium'], avg_tempo['high']]):
            plt.text(i, v, f'{v:.0f}', ha='center', va='bottom')
        
        plt.tight_layout(pad=3.0)
        return fig

    def prepare_data(self):
        """
        Prepara os dados para treinamento
        """
        if self.data is None:
            raise ValueError("Dados não coletados ainda.")
            
        feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode',
                         'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        
        X = self.data[feature_columns]
        y = self.data['popularity_category']
        
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train_model(self):
        """
        Treina o modelo e mostra resultados simplificados
        """
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        fig = plt.figure(figsize=(20, 8))
        
        features = ['Dançabilidade', 'Energia', 'Tom', 'Volume', 
                   'Modo', 'Vocais', 'Instrumental', 
                   'Ao Vivo', 'Positividade', 'Tempo']
        
        importances = pd.Series(
            self.model.feature_importances_,
            index=features
        ).sort_values(ascending=True)
        
        bars = plt.barh(range(len(importances)), importances, color='#1DB954')
        plt.yticks(range(len(importances)), importances.index)
        plt.title('Características que Mais Influenciam na Popularidade', fontsize=14, pad=20)
        
        for i, v in enumerate(importances):
            plt.text(v, i, f' {v:.1%}', va='center')
        
        plt.tight_layout(pad=3.0)
        return fig

    def generate_insights(self):
        """
        Gera insights principais dos dados
        """
        if self.data is None:
            return {}
            
        insights = {
            'tempo': self.data[self.data['popularity_category']=='high']['tempo'].mean(),
            'energia_popularidade': self.data['energy'].mean(),
            'caracteristica_dominante': self.data[['danceability', 'energy', 'valence']].mean().idxmax(),
            'distribuicao': (self.data['popularity_category']=='high').mean()*100
        }
        
        return insights
