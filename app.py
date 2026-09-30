from flask import Flask, send_from_directory
import os

app = Flask(__name__, static_folder='web', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

if __name__ == '__main__':
    # Run the server on 0.0.0.0 to be accessible outside the Docker container
    # Port 7001 is used as requested
    app.run(host='0.0.0.0', port=7001)
