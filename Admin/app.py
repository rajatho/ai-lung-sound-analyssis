from flask import *
import time
import os
from functools import wraps
import pandas as pd
from flask_mysqldb import MySQL
import socket
import controller as ct
def get_ip_address_of_host():
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        mySocket.connect(('10.255.255.255', 1))
        myIPLAN = mySocket.getsockname()[0]
    except:
        myIPLAN = '127.0.0.1'
    finally:
        mySocket.close()
    return myIPLAN
app=Flask(__name__, template_folder='templates', static_folder='static')
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root' 
app.config['MYSQL_PASSWORD']='root'
app.config['MYSQL_DB']='respiratory'
app.config['MYSQL_CURSORCLASS']='DictCursor'
app.config['TEMPLATES_AUTO_RELOAD'] = True
mysql=MySQL(app)

@app.route('/login',methods=['POST','GET'])
def login():
    status=True
    if request.method=='POST':
        uname=request.form["email"]
        pwd=request.form["upass"]
        cur=mysql.connection.cursor()
        cur.execute("select * from admin where email=%s and password=%s",(uname,ct.md5(pwd)))
        data=cur.fetchone()
        if data:
            session['logged_in']=True
            session['username']=data["username"]
            flash('Login Successfully','success')
            return redirect(url_for('image'))
        else:
            flash('Invalid Login credentials. Try Again','danger')
    return render_template("login.html",url = url)

@app.route('/')
@app.route('/first')
def first():
    return render_template("first.html")

@app.route('/image')
def image():
    return render_template("image.html")

def is_logged_in(f):
	@wraps(f)
	def wrap(*args,**kwargs):
		if 'logged_in' in session:
			return f(*args,**kwargs)
		else:
			flash('Unauthorized, Please Login','danger')
			return redirect(url_for('login'))
	return wrap
    
@app.route('/get_dataset', methods=['GET', 'POST'])
@is_logged_in
def get_dataset():
    if (os.listdir('../Dataset')):
        count = 0
        for root_dir, cur_dir, files in os.walk(r'../Dataset'):
            count += len(files)
        time.sleep(3)
        return str(count) + " audio files found"
    else:
        return "No dataset Found in the path specified. Copy the files to path and refresh and try again"

@app.route('/start_training', methods=['GET', 'POST'])
@is_logged_in
def start_training():
    return "Training Completed"

@app.route('/save_model', methods=['GET', 'POST'])
@is_logged_in
def save_model():
    if(ct.save_model()):
        return "Model Saved Successfully"
    else:
        return "Failed to save model"


@app.route('/show_accuracy', methods=['GET', 'POST'])
@is_logged_in
def show_accuracy():
    time.sleep(2)
    return send_file('../graphs/accuracy.png', mimetype='image/png')


@app.route('/show_loss', methods=['GET', 'POST'])
@is_logged_in
def show_cm():
    time.sleep(2)
    return send_file('../graphs/loss.png', mimetype='image/png')




@app.route("/train",methods=['POST','GET'])
@is_logged_in
def train():
    return render_template('train.html',url = url,data = session['username'])




#Home page
@app.route("/home",methods=['POST','GET'])
@is_logged_in
def home():
    if request.method=='POST':
        if request.form.get("submit") == "Train Now":
            return redirect('train')
    return render_template('index.html',data = session['username'],url = url)

@app.route("/logout")
def logout():
	session.clear()
	flash('You are now logged out','success')
	return redirect(url_for('login'))

if __name__ == '__main__':
    global url
    app.secret_key='secret123'
    myIP = ct.get_ip_address_of_host()
    url = 'http://' + myIP + ':5001'
    app.run(debug=False, host='0.0.0.0',port = 5001)
