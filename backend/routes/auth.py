from flask import Blueprint, request, jsonify
from datetime import datetime
from utils.data_handler import load_json, save_json, sync_users_to_csv, convert_list_to_dict
from utils.initializer import initialize_user_files
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.json
    
    # Validate fields
    required_fields = ['name', 'email', 'password', 'phone', 'age', 'gender', 'location']
    for field in required_fields:
        if field not in data or not str(data[field]).strip():
            return jsonify({'error': f'Missing or empty field: {field}'}), 400
            
    try:
        age_int = int(data['age'])
    except ValueError:
        return jsonify({'error': 'Age must be an integer'}), 400

    # Ensure data is parsed correctly from request
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request format, expected JSON dictionary.'}), 400

    # Load users.json specifically as a LIST
    users = load_json('users.json', default_value=[])
    if isinstance(users, dict):
        # Convert backward from dict -> list if corrupted
        users = list(users.values())

    email_key = data['email'].strip().lower()
    
    # Check duplicate email
    existing_user = next((u for u in users if isinstance(u, dict) and u.get('email', '').strip().lower() == email_key), None)
    if existing_user:
        return jsonify({'error': 'Email is already registered'}), 409
            
    # Assign next ID
    next_id = 1
    if users:
        next_id = max((u.get('id', 0) for u in users if isinstance(u, dict)), default=0) + 1
        
    # Create new user
    new_user = {
        "id": next_id,
        "name": data['name'].strip(),
        "email": email_key,
        "password": generate_password_hash(data['password']),
        "phone": data['phone'].strip(),
        "age": age_int,
        "gender": data['gender'].strip(),
        "location": data['location'].strip(),
        "traits": data.get('traits', []),
        "personality_type": data.get('personality_type', 'Unknown'),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    
    # Append to list and save
    users.append(new_user)
    save_json('users.json', users)
    
    # Initialize entries in ALL other JSON files
    initialize_user_files(email_key)
    
    # Call sync_users_to_csv()
    sync_users_to_csv()
    
    return jsonify({
        'message': 'Registration successful',
        'user': new_user
    }), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request format, expected JSON dictionary.'}), 400

    if 'email' not in data or 'password' not in data:
        return jsonify({'error': 'Email and password are required'}), 400
        
    users = load_json('users.json', default_value=[])
    if isinstance(users, dict):
        users = list(users.values())
        
    email_key = data['email'].strip().lower()
    
    user = next((u for u in users if isinstance(u, dict) and u.get('email', '').strip().lower() == email_key), None)
    
    if user:
        # Verify password if the user has one (backward compatibility for old users)
        if 'password' in user:
            if not check_password_hash(user['password'], data['password']):
                return jsonify({'error': 'Invalid credentials'}), 401
                
        return jsonify({
            'message': 'Login successful',
            'user': user
        }), 200
            
    return jsonify({'error': 'User not found'}), 404
