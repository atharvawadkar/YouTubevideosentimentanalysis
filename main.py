from flask import Flask, render_template, request, redirect, session
import mysql.connector
from sentiments import second

import os
 

 
 
app = Flask(__name__)
 
# initializing the user cookie
app.secret_key = os.urandom(24)
 
# blueprint to call the second python file in the project.
app.register_blueprint(second)
 
# establishing a connection with mysql database made in xampp
try:
    conn = mysql.connector.connect(
        host="localhost", user="root", password="anilvanita68$", database="sentiments")
    cursor = conn.cursor()
except:
    print("An exception occurred")




@app.route('/')
def login():
    return render_template('register.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('home.html')
    else:
        return redirect('/')


# @app.route('/login_validation', methods=['POST'])
# def login_validation():
#     email=request.form.get('email')
#     password=request.form.get('password')
    
#     cur=conn.cursor()
#     cur.execute("""SELECT * from `db_data` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(email, password))
#     users = cur.fetchall()
#     if len(users)>0:
#         session['user_id']=users[0][0]
#         return redirect('/home')
#     else:
#         return redirect('/login')


@app.route('/add_user', methods=['POST'])
def add_user():
   
    cur=conn.cursor()
    name=request.form.get('uname')
    
    cur.execute("""INSERT INTO `users` (`username`) VALUES ('{}')""".format(name))
    conn.commit()
    cur.execute("""SELECT * from `users` WHERE `username` LIKE '{}'""".format(name))
    myuser=cur.fetchall()
    session['user_id']=myuser[0][0]
    return redirect('/home')

@app.route('/logout')
def logout():
    session.pop('user_id')
    if os.path.isfile("static//images/plot1.png"):
        os.remove("static//images/plot1.png")
    return redirect('/')

if __name__=="__main__":
    app.run(debug=True)