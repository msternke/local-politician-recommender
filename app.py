from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/local-race')
def local_race():
    return render_template('local-race.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
