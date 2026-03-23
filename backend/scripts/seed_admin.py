import sys
import os
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from werkzeug.security import generate_password_hash
from utils.data_handler import load_json, save_json

def seed_admin():
    users_data = load_json('users.json', default_value={})
    
    from utils.data_handler import convert_list_to_dict
    if isinstance(users_data, list):
        users = convert_list_to_dict(users_data, key='email')
    else:
        users = users_data
        
    admin_email = "admin@cognitive.com"
    admin_password = "admin123" # Simple default password
    
    # Assign next ID
    next_id = 999999
    
    users[admin_email] = {
        "id": next_id,
        "name": "System Administrator",
        "email": admin_email,
        "password": generate_password_hash(admin_password),
        "role": "admin",
        "phone": "-",
        "age": 0,
        "gender": "Other",
        "location": "System",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
    
    save_json('users.json', users)
    print(f"Admin user seeded successfully: {admin_email} / {admin_password}")

if __name__ == "__main__":
    seed_admin()
