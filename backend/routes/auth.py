from flask import Blueprint, request, jsonify
from datetime import datetime
from utils.data_handler import load_json, save_json, sync_users_to_csv, convert_list_to_dict
from utils.initializer import initialize_user_files

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.json
    
    # Validate fields
    required_fields = ['name', 'email', 'phone', 'age', 'gender', 'location']
    for field in required_fields:
        if field not in data or not str(data[field]).strip():
            return jsonify({'error': f'Missing or empty field: {field}'}), 400
            
    try:
        age_int = int(data['age'])
    except ValueError:
        return jsonify({'error': 'Age must be an integer'}), 400

    users_data = load_json('users.json', default_value={})
    users = convert_list_to_dict(users_data, key='email')
    
    email_key = data['email'].strip().lower()
    # Check duplicate email
    if email_key in users:
        return jsonify({'error': 'Email is already registered'}), 409
            
    # Assign next ID
    next_id = 1
    if users:
        next_id = max(u.get('id', 0) for u in users.values()) + 1
        
    # Create new user
    new_user = {
        "id": next_id,
        "name": data['name'].strip(),
        "email": data['email'].strip().lower(),
        "phone": data['phone'].strip(),
        "age": age_int,
        "gender": data['gender'].strip(),
        "location": data['location'].strip(),
        "traits": data.get('traits', []),
        "personality_type": data.get('personality_type', 'Unknown'),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    
    # Add to users.json
    users[email_key] = new_user
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
    # Example login logic if needed
    data = request.json
    if 'email' not in data:
        return jsonify({'error': 'Email is required'}), 400
        
    users_data = load_json('users.json', default_value={})
    users = convert_list_to_dict(users_data, key='email')
    
    email_key = data['email'].strip().lower()
    if email_key in users:
        return jsonify({
            'message': 'Login successful',
            'user': users[email_key]
        }), 200
            
    return jsonify({'error': 'User not found'}), 404
