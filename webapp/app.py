from flask import Flask, jsonify, request, render_template
from flask_sqlalchemy import SQLAlchemy
from utils import singleimg_embedding, cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/test'
app.debug = True
db = SQLAlchemy(app)

upload_dir = os.path.dirname(os.path.abspath(__file__))
allowed_ext = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = upload_dir


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_ext


class MissingPerson(db.Model):
    __tablename__ = 'missing_person'
    id = db.Column(db.Integer(), primary_key=True)
    first_name = db.Column(db.String(255), nullable=False)
    last_name = db.Column(db.String(255), nullable=False)
    last_seen = db.Column(db.String(255), nullable=False)
    embedding = db.Column(db.String(255), nullable=False)

    def __init__(self, id, first_name, last_name, last_seen, embedding):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.last_seen = last_seen
        self.embedding = embedding


@app.route('/missingperson', methods=['GET'])
def get_missingperson():
    """
    Query result of all missing person in
    :return: JSON
    """
    all_missingpersons = MissingPerson.query.all()
    output = []
    for person in all_missingpersons:
        curr_result = {}
        print(person.id)
        curr_result['id'] = person.id
        curr_result['first_name'] = person.first_name
        curr_result['last_name'] = person.last_name
        curr_result['last_seen'] = person.last_seen
        curr_result['embedding'] = person.embedding
        output.append(curr_result)

    return jsonify(output)

# FLASK API Calls
@app.route('/missingperson', methods=['POST'])
def test():
    """
    API Call for adding data in DB
    """
    fetch_missingpersons = request.get_json()

    # Add data in the DB
    add_person = MissingPerson(id=fetch_missingpersons['id'], first_name=fetch_missingpersons[
        'first_name'], last_name=fetch_missingpersons['last_name'],
                               last_seen=fetch_missingpersons['last_seen'],
                               embedding=fetch_missingpersons['embedding'])
    db.session.add(add_person)
    db.session.commit()
    return jsonify(fetch_missingpersons)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    API call for HTML form to add data from the front end
    """
    if request.method == 'POST':
        user_details = request.form
        print(user_details)
        file = request.files['myfile']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = cv2.imread(filename)
        emb = singleimg_embedding(image)

        add_person = MissingPerson(id=user_details['id'], first_name=user_details[
            'first_name'], last_name=user_details['last_name'],
                                   last_seen=user_details['last_seen'],
                                   embedding=emb)
        print(add_person)
        db.session.add(add_person)
        db.session.commit()
        return 'Success'
    return render_template('index.html')


@app.route('/missingperson', methods=['PUT'])
def update_data():
    """
        API call for updating data
    """
    fetch_missingpersons = request.get_json()
    update_data = MissingPerson.query.filter_by(id=fetch_missingpersons['id']).first()
    update_data.embedding = fetch_missingpersons['embedding']
    db.session.commit()
    return jsonify(fetch_missingpersons)
