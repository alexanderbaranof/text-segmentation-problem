from flask import Flask, render_template, url_for, request, send_from_directory
import os
import predictor
app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/contacts")
def contacts():
    return render_template('contacts.html')

@app.route("/loaded", methods=["POST"])
def loaded():
    
    target = os.path.join(APP_ROOT, 'files/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".txt") or (ext == ".pdf"):
            print("File supported moving on...")
        else:
            pass
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
    label, intro_text, main_text, end_text = predictor.main()
    if label == 0:
    	language = 'русский'
    if label == 1:
    	language = 'английский'
    if label == 2:
    	language = 'украинский'



# return send_from_directory("images", filename, as_attachment=True)
    return render_template('loaded.html', lang = language, intro_text = intro_text, main_text = main_text, end_text = end_text )

if __name__ == '__main__':
    app.run(debug=True)
