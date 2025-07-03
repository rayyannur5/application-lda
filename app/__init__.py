# /app/__init__.py

from flask import Flask
from flask_socketio import SocketIO

# Inisialisasi SocketIO di scope global.
# async_mode='eventlet' dipilih karena performanya yang baik untuk SocketIO.
# Pastikan 'eventlet' sudah diinstal: pip install eventlet
socketio = SocketIO(async_mode='eventlet')

def create_app():
    """Factory function untuk membuat instance aplikasi Flask."""
    app = Flask(__name__)
    
    # Kunci rahasia sangat penting untuk mengamankan session pengguna.
    # Ganti dengan string acak dan sulit ditebak di lingkungan produksi.
    app.config['SECRET_KEY'] = 'kunci-rahasia-yang-sangat-aman-dan-unik'
    app.config['SERVER_NAME'] = '127.0.0.1:5000'

    # Daftarkan blueprint yang berisi rute-rute aplikasi.
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # Hubungkan instance SocketIO dengan aplikasi Flask.
    socketio.init_app(app)

    return app