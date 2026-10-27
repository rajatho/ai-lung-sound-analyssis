import os
import socket
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_from_directory, abort
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
import controller as ct

UPLOAD_FOLDER = os.path.join('static', 'uploads')
GRAPHS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'graphs')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}


def get_ip_address_of_host():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(('10.255.255.255', 1))
        return sock.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        sock.close()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'secret123'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'respiratory'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

mysql = MySQL(app)

output_file = None
url = ''


def get_db_cursor():
    return mysql.connection.cursor()


def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        flash('Please login to continue.', 'danger')
        return redirect(url_for('login'))
    return wrap


@app.route('/')
def root():
    if 'logged_in' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session:
        return redirect(url_for('home'))

    error_msg = None
    active_tab = request.args.get('tab', 'login')

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        pwd = request.form.get('upass', '')

        if not email or not pwd:
            error_msg = 'Please enter email and password.'
        else:
            try:
                cur = get_db_cursor()
                cur.execute(
                    'SELECT * FROM user WHERE email=%s AND password=%s',
                    (email, pwd)
                )
                data = cur.fetchone()
                cur.close()
                if data:
                    session['logged_in'] = True
                    session['username'] = data['username']
                    flash('Login successful!', 'success')
                    return redirect(url_for('home'))
                error_msg = 'Invalid email or password.'
            except Exception as exc:
                error_msg = f'Database error: {exc}'

    return render_template(
        'login.html', url=url, active_tab=active_tab, error=error_msg
    )


@app.route('/reg', methods=['GET', 'POST'])
def reg():
    if 'logged_in' in session:
        return redirect(url_for('home'))

    error_msg = None
    success_msg = None

    if request.method == 'POST':
        name = request.form.get('uname', '').strip()
        email = request.form.get('email', '').strip()
        pwd = request.form.get('upass', '')

        if not name or not email or not pwd:
            error_msg = 'Please fill in all fields.'
        else:
            try:
                cur = get_db_cursor()
                cur.execute(
                    'SELECT * FROM user WHERE username=%s OR email=%s',
                    (name, email)
                )
                data = cur.fetchone()
                if data:
                    error_msg = 'Username or email already exists.'
                else:
                    cur.execute(
                        'INSERT INTO user(username, password, email) VALUES(%s, %s, %s)',
                        (name, pwd, email)
                    )
                    mysql.connection.commit()
                    session['logged_in'] = True
                    session['username'] = name
                    flash('Registration successful! Welcome.', 'success')
                    cur.close()
                    return redirect(url_for('home'))
                cur.close()
            except Exception as exc:
                error_msg = f'Database error: {exc}'

        return render_template(
            'login.html', url=url, active_tab='register',
            error=error_msg, success=success_msg
        )

    return redirect(url_for('login', tab='register'))


@app.route('/graphs/<path:filename>')
def serve_training_graph(filename):
    if filename not in ('accuracy.png', 'loss.png'):
        abort(404)
    return send_from_directory(GRAPHS_FOLDER, filename)


@app.route('/home')
@is_logged_in
def home():
    graphs = ct.training_graph_paths()
    return render_template(
        'index.html', url=url, username=session.get('username'),
        error=None, training_graphs=graphs
    )


@app.route('/logout')
def logout():
    global output_file
    output_file = None
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


@app.route('/predict', methods=['POST'])
@is_logged_in
def predict():
    global output_file

    if 'file' not in request.files:
        return render_template(
            'index.html', url=url, username=session.get('username'),
            error='No file selected.', training_graphs=ct.training_graph_paths()
        )

    uploaded = request.files['file']
    if uploaded.filename == '':
        return render_template(
            'index.html', url=url, username=session.get('username'),
            error='No file selected.', training_graphs=ct.training_graph_paths()
        )

    if not allowed_file(uploaded.filename):
        return render_template(
            'index.html', url=url, username=session.get('username'),
            error='Unsupported format. Please upload WAV, MP3, OGG, or FLAC.',
            training_graphs=ct.training_graph_paths()
        )

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    safe_name = secure_filename(uploaded.filename)
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    uploaded.save(file_path)
    output_file = file_path

    try:
        result = ct.predict(file_path)
    except Exception as exc:
        return render_template(
            'index.html', url=url, username=session.get('username'),
            error=str(exc),
            training_graphs=ct.training_graph_paths()
        )

    session['prediction'] = {
        'detections': result['detections'],
        'probabilities': result['probabilities'],
        'labels': result['labels'],
        'dual': result['dual'],
        'model_accuracy': result['accuracy'],
        'spectrogram': result.get('spectrogram'),
    }
    session['audio_file'] = file_path

    if result['dual']:
        return redirect(url_for('reports_hub'))
    return redirect(url_for('report_page', index=1))


@app.route('/reports')
@is_logged_in
def reports_hub():
    prediction = session.get('prediction')
    if not prediction or not prediction.get('dual'):
        return redirect(url_for('home'))

    return render_template(
        'reports_hub.html',
        url=url,
        username=session.get('username'),
        prediction=prediction,
        filename=session.get('audio_file'),
    )


@app.route('/reports/<int:index>')
@is_logged_in
def report_page(index):
    prediction = session.get('prediction')
    if not prediction:
        flash('No analysis found. Please upload audio first.', 'danger')
        return redirect(url_for('home'))

    detections = prediction.get('detections', [])
    if index < 1 or index > len(detections):
        return redirect(url_for('home'))

    return render_template(
        'report.html',
        url=url,
        username=session.get('username'),
        detection=detections[index - 1],
        prediction=prediction,
        report_index=index,
        total_reports=len(detections),
        filename=session.get('audio_file'),
    )


@app.route('/get_result', methods=['GET'])
@is_logged_in
def get_result():
    if not output_file:
        return jsonify({'error': 'No audio uploaded yet.'}), 400
    return jsonify(ct.predict(output_file))


if __name__ == '__main__':
    url = 'http://' + get_ip_address_of_host() + ':5000'
    print('Loading AI model (one-time startup)...')
    ct.load_classifier()
    print('Server ready at', url)
    app.run(debug=False, host='0.0.0.0', port=5000)
