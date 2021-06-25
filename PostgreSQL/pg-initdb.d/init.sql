CREATE SEQUENCE missing_person_seq;

CREATE TABLE IF NOT EXISTS missing_person(
id INTEGER NOT NULL PRIMARY KEY DEFAULT nextval('missing_person_seq'),
first_name VARCHAR(255) NOT NULL,
last_name VARCHAR(255) NOT NULL,
last_seen VARCHAR(255) NOT NULL,
embedding  VARCHAR(255)  NULL
);

ALTER SEQUENCE missing_person_seq OWNED BY missing_person.id;


