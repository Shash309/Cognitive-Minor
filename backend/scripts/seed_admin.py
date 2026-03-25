"""
Seed script — creates the admin account inside users.json.
Run once:  python scripts/seed_admin.py

Environment variables (optional):
  ADMIN_NAME      default "Platform Admin"
  ADMIN_EMAIL     default "admin@cognitive.ai"
  ADMIN_PASSWORD  default "Admin@2026"
"""

import os
import sys
import json

# Ensure project root is on sys.path so utils.data_handler can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from werkzeug.security import generate_password_hash
from utils.data_handler import load_json, save_json

ADMIN_NAME     = os.environ.get('ADMIN_NAME', 'Platform Admin')
ADMIN_EMAIL    = os.environ.get('ADMIN_EMAIL', 'admin@cognitive.ai')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'Admin@2026')


def seed():
    users = load_json('users.json', default_value=[])
    if isinstance(users, dict):
        users = list(users.values())

    # Check if admin already exists
    existing = next(
        (u for u in users
         if isinstance(u, dict) and u.get('email', '').lower() == ADMIN_EMAIL.lower()),
        None,
    )
    if existing:
        print(f"✅  Admin account already exists: {ADMIN_EMAIL}")
        return

    # Determine next ID
    max_id = max((u.get('id', 0) for u in users if isinstance(u, dict)), default=0)

    admin_user = {
        'id': max_id + 1,
        'name': ADMIN_NAME,
        'email': ADMIN_EMAIL.lower(),
        'password': generate_password_hash(ADMIN_PASSWORD),
        'phone': '',
        'age': 0,
        'gender': '',
        'location': '',
        'role': 'admin',
        'created_at': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
    }

    users.append(admin_user)
    save_json('users.json', users)
    print(f"🔐  Admin account created successfully!")
    print(f"    Email   : {ADMIN_EMAIL}")
    print(f"    Password: {ADMIN_PASSWORD}")
    print(f"    ⚠️  Change these credentials in production via environment variables.")


if __name__ == '__main__':
    seed()
