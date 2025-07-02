from flask import Blueprint, request, render_template, session, redirect, url_for
import time
from .controllers.predict import main_process

main = Blueprint('main', __name__)

@main.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == '1234' and password == '1234':
            session['user'] = username  # Simpan username ke session
            return {
                'message': 'success',
                'data': {
                    'nextRoute': '/'
                }
            }, 200
        else :
            return {
                'message' : 'username dan password salah',
            }, 400
    
    else :
        return render_template('login.html', title="Login Page")


@main.route('/logout')
def logout():
    session.pop('user', None)  # Hapus user dari session
    return redirect(url_for('main.login'))

@main.route('/')
def home():
    if 'user' in session:
        return render_template('home.html', title="Home Page")
    else:
        return redirect(url_for('main.login'))

@main.route('/proses', methods=['POST'])
def proses():
    if 'user' not in session:
        return redirect(url_for('main.login'))
    

    if 'file' not in request.files:
        return {
            'message': 'file not found'
        }, 400

    request.files['file'].save('./temp.csv')

    data = main_process('./temp.csv')

    return render_template('result.html', lda=data['html'], sentiment=data['sentiment'], kebijakan=data['kebijakan'])
