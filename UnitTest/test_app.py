import requests


response = requests.get('http://127.0.0.1:5000/missingperson')
print(response.status_code)
print(response.json())

# POST Data
data = {'embedding': None, 'first_name': 'Susan', 'id': 3, 'last_name': 'Pappachan', 'last_seen':
    'Seen in Mumbai'}
r = requests.post('http://127.0.0.1:5000/missingperson', json=data)
print(r.status_code)

# PUT Data
data = {'embedding': 'test',  'id': 1}
p = requests.put('http://127.0.0.1:5000/missingperson', json=data)
print(p.status_code)


