from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import psycopg2

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/test'
app.debug = True
db = SQLAlchemy(app)


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


@app.route('/missingperson', methods=['POST'])
def test():
    fetch_missingpersons = request.get_json()
    print(fetch_missingpersons)
    add_person = MissingPerson(id=fetch_missingpersons['id'], first_name=fetch_missingpersons[
        'first_name'], last_name=fetch_missingpersons['last_name'],
                               last_seen=fetch_missingpersons['last_seen'],
                               embedding=fetch_missingpersons['embedding'])
    db.session.add(add_person)
    db.session.commit()
    return jsonify(fetch_missingpersons)


@app.route('/missingperson', methods=['PUT'])
def update_data():
    fetch_missingpersons = request.get_json()
    update_data = MissingPerson.query.filter_by(id=fetch_missingpersons['id']).first()
    update_data.embedding = fetch_missingpersons['embedding']
    db.session.commit()
    return jsonify(fetch_missingpersons)
