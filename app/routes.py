# /app/routes.py

# 1. Tambahkan 'current_app' ke dalam import dari flask
from flask import Blueprint, request, render_template, session, redirect, url_for, current_app
import os
import uuid

from . import socketio
from .controllers.predict import main_process

main = Blueprint('main', __name__)

# 2. Ubah definisi wrapper untuk menerima 'app'
def background_task_wrapper(filepath, app):
    """
    Fungsi pembungkus yang sekarang berjalan di dalam application context.
    """
    # 3. Gunakan 'with app.app_context():' untuk membuat context
    with app.app_context():
        print(f"BACKGROUND WRAPPER: Memulai proses untuk {filepath} di dalam context.")
        
        def emit_status_callback(event, data):
            """Callback untuk mengirim update status."""
            socketio.emit(event, data)
            socketio.sleep(0) 

        try:
            # Panggil main_process, yang akan mengembalikan dictionary hasil
            hasil_dict = main_process(filepath=filepath, status_callback=emit_status_callback)

            top_key = max(hasil_dict['kebijakan'], key=hasil_dict['kebijakan'].get)
            
            # Sekarang render_template akan berhasil karena berada di dalam context
            result_html = render_template(
                'result.html', 
                lda=hasil_dict['html'], 
                sentiment=hasil_dict['sentiment'], 
                kebijakan=hasil_dict['kebijakan'],
                kebijakan_teratas=top_key
            )
            
            # Kirim string HTML yang sudah jadi ini ke client
            socketio.emit('proses_selesai', {'html': result_html, 'generated_analysis': hasil_dict['generated_analysis'], 'df_hasil_head':hasil_dict['df_hasil_head']})
            print(f"BACKGROUND WRAPPER: Proses selesai. String HTML dari result.html telah dikirim.")

        except Exception as e:
            error_message = f"❌ Terjadi error fatal di background: {str(e)}"
            print(error_message)
            socketio.emit('update_status', {'message': error_message})
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"BACKGROUND WRAPPER: File sementara {filepath} dihapus.")


@main.route('/proses', methods=['POST'])
def proses_upload():
    if 'user' not in session: return redirect(url_for('main.login'))
    if 'file' not in request.files or request.files['file'].filename == '':
        return {'message': 'File tidak valid atau tidak dipilih.'}, 400

    file = request.files['file']
    unique_filename = f"temp_{uuid.uuid4().hex}.csv"
    filepath = os.path.join('./', unique_filename)
    file.save(filepath)

    # 4. Dapatkan object aplikasi saat ini dan kirimkan ke background task
    app = current_app._get_current_object()
    socketio.start_background_task(target=background_task_wrapper, filepath=filepath, app=app)
    
    return {'message': '✅ File diterima! Proses dimulai...'}, 200

# Rute-rute dan handler lainnya tidak perlu diubah
@main.route('/login', methods=['POST','GET'])
def login():
    if request.method == 'POST':
        username = request.form.get('username'); password = request.form.get('password')
        if username == '1234' and password == '1234':
            session['user'] = username
            return {'message': 'success', 'data': {'nextRoute': '/'}}, 200
        else: return {'message': 'username dan password salah'}, 400
    return render_template('login.html', title="Login Page")

@main.route('/logout')
def logout():
    session.pop('user', None); return redirect(url_for('main.login'))

@main.route('/')
def home():
    if 'user' in session: return render_template('home.html', title="Home Page")
    else: return redirect(url_for('main.login'))

@socketio.on('connect')
def handle_connect(): print(f"Client terhubung: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect(): print(f"Client terputus: {request.sid}")