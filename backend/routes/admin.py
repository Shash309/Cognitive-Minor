import os
import jwt
import datetime
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# Since utils.data_handler seems to be used across the app
from utils.data_handler import load_json

SECRET_KEY = os.environ.get('SECRET_KEY', 'default-admin-secret-key')

admin_bp = Blueprint('admin', __name__)

def isAdmin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Token is missing or invalid format'}), 403
            
        try:
            token = auth_header.split(' ')[1]
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            if data.get('role') != 'admin':
                return jsonify({'error': 'Unauthorized: Not an admin'}), 403
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 403
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 403
            
        return f(*args, **kwargs)
    return decorated

@admin_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({'error': 'Email and password required'}), 400
        
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    users_data = load_json('users.json', default_value={})
    
    # Backward compatibility with dict or list
    from utils.data_handler import convert_list_to_dict
    if isinstance(users_data, list):
        users = convert_list_to_dict(users_data, key='email')
    else:
        users = users_data

    user = users.get(email)
    
    if not user or user.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized: Invalid credentials or role'}), 403
        
    user_password_hash = user.get('password', '')
    if not user_password_hash or not check_password_hash(user_password_hash, password):
        return jsonify({'error': 'Unauthorized: Invalid credentials or role'}), 403

    # Valid admin logic
    token = jwt.encode({
        'email': email,
        'role': 'admin',
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm="HS256")
    
    return jsonify({
        'message': 'Admin login successful',
        'token': token,
        'user': {
            'email': email,
            'name': user.get('name', 'Admin'),
            'role': 'admin'
        }
    }), 200

@admin_bp.route('/stats', methods=['GET'])
@isAdmin
def get_stats():
    users_data = load_json('users.json', default_value={})
    quiz_data = load_json('quiz_history.json', default_value={})
    voice_data = load_json('voice_history.json', default_value={})
    
    if isinstance(users_data, list):
        from utils.data_handler import convert_list_to_dict
        users = convert_list_to_dict(users_data, key='email')
    else:
        users = users_data

    total_users = 0
    active_users = 0
    for u in users.values():
        if u.get('role') != 'admin':
            total_users += 1
            # Very basic active user logic based on last_login existence
            if u.get('last_login'):
                active_users += 1

    total_quiz_attempts = 0
    for email, attempts in quiz_data.items():
        if isinstance(attempts, list):
            total_quiz_attempts += len(attempts)
        elif isinstance(attempts, dict) and "quiz_scores" in attempts:
            total_quiz_attempts += 1
            
    total_voice_analyses = 0
    for email, analyses in voice_data.items():
        if isinstance(analyses, list):
            total_voice_analyses += len(analyses)
        elif isinstance(analyses, dict):
            total_voice_analyses += 1

    return jsonify({
        'total_users': total_users,
        'active_users': active_users,
        'total_quiz_attempts': total_quiz_attempts,
        'total_voice_analyses': total_voice_analyses
    }), 200

@admin_bp.route('/users', methods=['GET'])
@isAdmin
def get_all_users():
    users_data = load_json('users.json', default_value={})
    if isinstance(users_data, list):
        from utils.data_handler import convert_list_to_dict
        users = convert_list_to_dict(users_data, key='email')
    else:
        users = users_data
        
    user_list = []
    for email, u in users.items():
        if u.get('role') != 'admin':
            user_list.append({
                'id': u.get('id'),
                'name': u.get('name', 'Unknown'),
                'email': u.get('email', email),
                'phone': u.get('phone', 'N/A'),
                'age': u.get('age', 'N/A'),
                'gender': u.get('gender', 'N/A'),
                'location': u.get('location', 'N/A'),
                'created_at': u.get('created_at', None),
                'last_login': u.get('last_login', None)
            })
            
    # sort by created_at descending
    user_list.sort(key=lambda x: x.get('created_at') or '', reverse=True)
    return jsonify(user_list), 200

@admin_bp.route('/user/<identifier>', methods=['GET'])
@isAdmin
def get_user_detail(identifier):
    users_data = load_json('users.json', default_value={})
    if isinstance(users_data, list):
        from utils.data_handler import convert_list_to_dict
        users = convert_list_to_dict(users_data, key='email')
    else:
        users = users_data
        
    target_user = None
    target_email = None
    
    if '@' in identifier:
        target_user = users.get(identifier.lower())
        target_email = identifier.lower()
    else:
        for email, u in users.items():
            if str(u.get('id')) == identifier:
                target_user = u
                target_email = email
                break
                
    if not target_user or target_user.get('role') == 'admin':
        return jsonify({'error': 'User not found'}), 404
        
    quiz_data = load_json('quiz_history.json', default_value={})
    voice_data = load_json('voice_history.json', default_value={})
    fused_data = load_json('career_fused_results.json', default_value={})
    
    user_quiz = quiz_data.get(target_email, [])
    user_voice = voice_data.get(target_email, [])
    user_fused = fused_data.get(target_email, {})
    
    return jsonify({
        'user': {
            'id': target_user.get('id'),
            'name': target_user.get('name'),
            'email': target_user.get('email'),
            'phone': target_user.get('phone'),
            'age': target_user.get('age'),
            'gender': target_user.get('gender'),
            'location': target_user.get('location'),
            'created_at': target_user.get('created_at'),
            'last_login': target_user.get('last_login')
        },
        'quiz_history': user_quiz,
        'voice_history': user_voice,
        'career_recommendations': user_fused
    }), 200

@admin_bp.route('/activity', methods=['GET'])
@isAdmin
def get_activity():
    users_data = load_json('users.json', default_value={})
    if isinstance(users_data, list):
        from utils.data_handler import convert_list_to_dict
        users = convert_list_to_dict(users_data, key='email')
    else:
        users = users_data
        
    activities = []
    for email, u in users.items():
        if u.get('role') != 'admin' and u.get('created_at'):
            activities.append({
                'id': str(u.get('id')) + '_reg',
                'type': 'registration',
                'user': u.get('name', email),
                'email': email,
                'timestamp': u.get('created_at')
            })
        if u.get('role') != 'admin' and u.get('last_login'):
            activities.append({
                'id': str(u.get('id')) + '_log',
                'type': 'login',
                'user': u.get('name', email),
                'email': email,
                'timestamp': u.get('last_login')
            })
            
    activities.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return jsonify(activities[:100]), 200
