from flask import Flask, render_template, send_from_directory
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("reinforced.html")

@app.route('/minimax')
def minimax():
    return render_template('minimax-alpha-beta.html')

@app.route('/random') 
def random():
    return render_template('random.html')

@app.route('/img/<path:path>')
def send_js(path):
    return send_from_directory('static/img', path)
if __name__ == '__main__':
   app.run()