"""
Admin-only routes — completely isolated from user authentication.
Uses itsdangerous token signing (no extra JWT dependency needed).
All endpoints protected by the @is_admin decorator.
"""

import os
import csv
import io
from functools import wraps
from flask import Blueprint, request, jsonify, current_app, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from datetime import datetime

from utils.data_handler import load_json, save_json

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# ─── Token helpers ───────────────────────────────────────────────
ADMIN_TOKEN_MAX_AGE = 43200  # 12 hours

def _get_serializer():
    secret = os.environ.get('ADMIN_SECRET_KEY', 'cognitive-admin-secret-2026')
    return URLSafeTimedSerializer(secret)


def is_admin(f):
    """Decorator — blocks every request that lacks a valid admin bearer token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized — token missing'}), 403

        token = auth.split(' ', 1)[1]
        try:
            data = _get_serializer().loads(token, max_age=ADMIN_TOKEN_MAX_AGE)
            if data.get('role') != 'admin':
                return jsonify({'error': 'Forbidden — not admin role'}), 403
            request.admin_email = data.get('email')
        except SignatureExpired:
            return jsonify({'error': 'Token expired — please log in again'}), 403
        except BadSignature:
            return jsonify({'error': 'Invalid token'}), 403

        return f(*args, **kwargs)
    return decorated


# ─── Admin Login ─────────────────────────────────────────────────
@admin_bp.route('/login', methods=['POST'])
def admin_login():
    body = request.get_json(silent=True) or {}
    email = (body.get('email') or '').strip().lower()
    password = body.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    users = load_json('users.json', default_value=[])
    if isinstance(users, dict):
        users = list(users.values())

    admin_user = next(
        (u for u in users
         if isinstance(u, dict)
         and u.get('email', '').strip().lower() == email
         and u.get('role') == 'admin'),
        None,
    )

    if not admin_user or not admin_user.get('password'):
        return jsonify({'error': 'Invalid credentials or not an admin'}), 403

    if not check_password_hash(admin_user['password'], password):
        return jsonify({'error': 'Invalid credentials or not an admin'}), 403

    token = _get_serializer().dumps({'email': email, 'role': 'admin'})
    safe_admin = {k: v for k, v in admin_user.items() if k != 'password'}

    return jsonify({
        'message': 'Admin login successful',
        'token': token,
        'admin': safe_admin,
    }), 200


# ─── GET /admin/stats ────────────────────────────────────────────
@admin_bp.route('/stats', methods=['GET'])
@is_admin
def admin_stats():
    users = load_json('users.json', default_value=[])
    if isinstance(users, dict):
        users = list(users.values())

    quiz_history = load_json('quiz_history.json', default_value={})
    voice_history = load_json('voice_history.json', default_value={})
    psych_profiles = load_json('psych_profiles.json', default_value={})
    career_fused = load_json('career_fused_results.json', default_value={})

    non_admin = [u for u in users if isinstance(u, dict) and u.get('role', 'user') != 'admin']
    total_users = len(non_admin)

    total_quizzes = sum(
        len(v) for v in quiz_history.values() if isinstance(v, list)
    )
    total_voice = sum(
        len(v) for v in voice_history.values() if isinstance(v, list)
    )
    total_psych = sum(
        1 for v in psych_profiles.values() if v
    )
    total_career = sum(
        1 for v in career_fused.values() if v
    )

    return jsonify({
        'total_users': total_users,
        'active_users': total_users,  # simplistic — all registered = active
        'total_quiz_attempts': total_quizzes,
        'total_voice_analyses': total_voice,
        'total_psych_assessments': total_psych,
        'total_career_recommendations': total_career,
    }), 200


# ─── GET /admin/users ────────────────────────────────────────────
@admin_bp.route('/users', methods=['GET'])
@is_admin
def admin_users():
    users = load_json('users.json', default_value=[])
    if isinstance(users, dict):
        users = list(users.values())

    result = []
    for u in users:
        if not isinstance(u, dict) or u.get('role') == 'admin':
            continue
        safe = {k: v for k, v in u.items() if k != 'password'}
        result.append(safe)

    return jsonify(result), 200


# ─── GET /admin/user/<id> ────────────────────────────────────────
@admin_bp.route('/user/<uid>', methods=['GET'])
@is_admin
def admin_user_detail(uid):
    try:
        user_id = int(uid)
    except ValueError:
        return jsonify({'error': 'Invalid user ID'}), 400

    users = load_json('users.json', default_value=[])
    if isinstance(users, dict):
        users = list(users.values())

    user = next(
        (u for u in users if isinstance(u, dict) and u.get('id') == user_id),
        None,
    )
    if not user:
        return jsonify({'error': 'User not found'}), 404

    email = user.get('email', '')
    quiz_h = load_json('quiz_history.json', default_value={}).get(email, [])
    voice_h = load_json('voice_history.json', default_value={}).get(email, [])
    psych = load_json('psych_profiles.json', default_value={}).get(email)
    career = load_json('career_fused_results.json', default_value={}).get(email)
    progress = load_json('user_progress.json', default_value={}).get(email)

    safe = {k: v for k, v in user.items() if k != 'password'}

    return jsonify({
        'user': safe,
        'quiz_history': quiz_h,
        'voice_history': voice_h,
        'psych_profile': psych,
        'career_fused': career,
        'progress': progress,
    }), 200


# ─── GET /admin/activity ─────────────────────────────────────────
@admin_bp.route('/activity', methods=['GET'])
@is_admin
def admin_activity():
    users = load_json('users.json', default_value=[])
    if isinstance(users, dict):
        users = list(users.values())

    logs = []
    for u in users:
        if not isinstance(u, dict) or u.get('role') == 'admin':
            continue
        logs.append({
            'type': 'registration',
            'email': u.get('email'),
            'name': u.get('name', 'Unknown'),
            'timestamp': u.get('created_at', 'Unknown'),
        })

    # Sort by timestamp descending; protect against non-string timestamps
    logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return jsonify(logs[:100]), 200


# ─── GET /admin/export/users  (CSV download) ─────────────────────
@admin_bp.route('/export/users', methods=['GET'])
@is_admin
def export_users_csv():
    users = load_json('users.json', default_value=[])
    if isinstance(users, dict):
        users = list(users.values())

    fields = ['id', 'name', 'email', 'phone', 'age', 'gender', 'location', 'created_at']
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for u in users:
        if not isinstance(u, dict) or u.get('role') == 'admin':
            continue
        writer.writerow({k: u.get(k, '') for k in fields})

    resp = make_response(buf.getvalue())
    resp.headers['Content-Type'] = 'text/csv'
    resp.headers['Content-Disposition'] = 'attachment; filename=users_export.csv'
    return resp
