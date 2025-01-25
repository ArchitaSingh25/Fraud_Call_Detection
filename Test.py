from flask import Flask,request,render_template

app=Flask(__name__)

@app.route('/Fraud Call Detection')
def index():
    return render_template("index.html")


