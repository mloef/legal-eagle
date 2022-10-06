from flask import Flask
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
from model import index_flow, run_query, preprocess, highlight
import time
import pickle

UPLOAD_FOLDER = 'uploads'
EMBEDDINGS_FOLDER = 'embeddings'
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#TODO: class instead of global
index = {}

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                content = preprocess(str(file.stream.read()))
                with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), "w") as fp:
                    fp.write(content)

                embeddings = index_flow(content)
                with open(os.path.join(EMBEDDINGS_FOLDER, filename), "wb") as fp:
                    pickle.dump(embeddings, fp)

                index[filename] = embeddings

        return '<p>Successfully uploaded files</p>'


@app.route('/form')
def form():
    return '''<!DOCTYPE html>
<form action="/data" method = "POST">
    <p>Text input <input type = "text" name = "Input Field" /></p>
    <p><input type = "submit" value = "submit" /></p>
</form>'''


@app.route('/data/', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form['Input Field']
        filename = "upload" + str(int(time.time())) + ".txt"
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), "w") as fp:
            fp.write(form_data)

        embeddings = index_flow(form_data)

        with open(os.path.join(EMBEDDINGS_FOLDER, filename), "wb") as fp:
            pickle.dump(embeddings, fp)

        index[filename] = embeddings
        return "<p>Successfully saved text</p>"


def update_index():
    global index
    for filename in os.listdir(EMBEDDINGS_FOLDER):
        path = os.path.join(EMBEDDINGS_FOLDER, filename)
        if os.path.isfile(path):
            with open(path, 'rb') as fp:
                embedding = pickle.load(fp)
                index[filename] = embedding  #TODO:class instead of global


@app.route('/query')
def query():
    return render_template('query.html')


@app.route('/query-result/', methods=['POST', 'GET'])
def query_result():
    if request.method == 'GET':
        return f"The URL /query-result is accessed directly. Try going to '/query' to submit query"
    if request.method == 'POST':
        query = request.form['Input Field']
        result = run_query(query, index) #TODO:configurable k, maybe move index into model.py?

        filenames = [res[2] for res in result]

        files = {}
        for filename in filenames:
            path = os.path.join(UPLOAD_FOLDER, filename)
            with open(path, 'rb') as fp:
                files[filename] = fp.read()

        passages = {}
        for res in result: #TODO:result as a namedtuple
            filename = res[2]
            start, end = res[1]
            score = res[0]
            passage = files[filename][start:end].decode("utf-8")
            _, (highlight_start, highlight_end), _ = highlight(query, passage)[0]
            passage = (passage[0:highlight_start], passage[highlight_start:highlight_end], passage[highlight_end:-1])
            passages[passage] = float(score[0]) #TODO:passage collisions?

        dump_query(query, [float(res[0][0]) for res in result])

        return render_template('query-result.html', result=passages)


def show_index(): #TEST ONLY
    for filename, embeddings in index.items():
        path = os.path.join(UPLOAD_FOLDER, filename)
        with open(path, 'rb') as fp:
            content = fp.read()
            for embedding, location in embeddings:
                start, end = location
                passage = content[start:end]
                print(filename, location, passage)#, embedding)


def dump_query(query, scores): #TODO: keep queries in memory, write periodically and on shutdown
    filename = Path('queries.data')
    filename.touch(exist_ok=True)
    with open(filename, 'rb+') as fp:
        content = fp.read()
        if len(content):
            queries = pickle.loads(content)
        else:
            queries = {}

        queries[query] = scores

    with open(filename, 'wb+') as fp:
        pickle.dump(queries, fp)


if __name__ == "__main__":
    update_index()
    #show_index()
    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)
