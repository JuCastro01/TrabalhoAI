from flask import Flask, render_template, jsonify
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from spotifyml import SpotifyMLPipeline

app = Flask(__name__)

# Credenciais do Spotify
CLIENT_ID = "ddcd98f76a6a438799e331587a467a27"
CLIENT_SECRET = "c427a53bb4ee42cca45b1fd5ef263cb0"

# Inicialização global do pipeline
pipeline = None

@app.route('/')
def home():
    """Rota principal que renderiza a interface"""
    return render_template('index.html')

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Executa a análise completa dos dados"""
    global pipeline
    
    try:
        # Inicializa o pipeline
        pipeline = SpotifyMLPipeline(CLIENT_ID, CLIENT_SECRET)
        
        # Playlists para análise
        playlist_ids = [
            "37i9dQZF1DXcBWIGoYBM5M",  # Today's Top Hits
            "37i9dQZF1DX0XUsuxWHRQd",  # RapCaviar
            "37i9dQZF1DX4JAvHpjipBk", # New Music Friday
            "37i9dQZF1DX0Yxoavh5qJV",
            "4rnleEAOdmFAbRcNCgZMpY"
        ]
        
        # Coleta dados
        pipeline.collect_data(playlist_ids)
        
        # Dicionário para armazenar as imagens dos gráficos
        figures = {}
        
        # 1. Análise de Distribuição
        plt.close('all')
        fig = pipeline.visualize_data_distribution()
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        figures['distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # 2. Análise do Modelo
        plt.close('all')
        fig = pipeline.train_model()
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        figures['model'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Gera insights
        insights = pipeline.generate_insights()
        
        # Formata os insights para melhor apresentação
        formatted_insights = {
            'tempo': f"{float(insights['tempo']):.0f}",
            'energia_popularidade': f"{float(insights['energia_popularidade']):.2f}",
            'caracteristica_dominante': insights['caracteristica_dominante'].title(),
            'distribuicao': f"{float(insights['distribuicao']):.1f}%"
        }
        
        return jsonify({
            'success': True,
            'figures': figures,
            'insights': formatted_insights
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.errorhandler(404)
def page_not_found(e):
    """Handler para páginas não encontradas"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handler para erros do servidor"""
    return jsonify({
        'success': False,
        'error': 'Erro interno do servidor. Por favor, tente novamente.'
    }), 500

if __name__ == '__main__':
    app.run(debug=True)