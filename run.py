# /run.py

from app import create_app, socketio

# Membuat instance aplikasi dari factory function
app = create_app()

if __name__ == '__main__':
    # Jalankan aplikasi menggunakan server SocketIO dengan dukungan asynchronous dari eventlet.
    # Ini penting agar WebSocket dapat berjalan dengan baik.
    # Jangan gunakan 'flask run' untuk menjalankan aplikasi ini.
    print("Starting Flask-SocketIO server with eventlet...")
    socketio.run(app, debug=True)