import json
import os
import csv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def get_file_path(filename):
    return os.path.join(DATA_DIR, filename)

def convert_list_to_dict(data, key="email"):
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        print("⚠️ Converted list-based DB to dict-based DB")
        converted = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            item_key = item.get(key)
            if not item_key:
                print(f"⚠️ Validation Error: Entry missing '{key}', skipping.")
                continue
            if item_key in converted:
                print(f"⚠️ Validation Warning: Duplicate '{key}' found: {item_key}. Overwriting previous entry.")
            converted[item_key] = item
        return converted
    
    print("⚠️ Data is not a list or dict. Returning empty dict.")
    return {}

def safe_get(db, email):
    if db is None:
        return None
    if isinstance(db, dict):
        return db.get(email)
    if isinstance(db, list):
        for item in db:
            if isinstance(item, dict) and item.get("email") == email:
                return item
    return None

def load_json(filename, default_value=None):
    if default_value is None:
        default_value = []
    
    filepath = get_file_path(filename)
    if not os.path.exists(filepath):
        save_json(filename, default_value)
        return default_value
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        save_json(filename, default_value)
        return default_value
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return default_value

def save_json(filename, data):
    filepath = get_file_path(filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def append_entry(filename, entry):
    data = load_json(filename, default_value=[])
    if not isinstance(data, list):
        print(f"Warning: {filename} was not a list. Resetting to empty list for consistent schema.")
        data = []
    data.append(entry)
    save_json(filename, data)

def sync_users_to_csv():
    users_data = load_json('users.json', default_value={})
    users_data = convert_list_to_dict(users_data, key='email')
    users = list(users_data.values())
    
    if not users:
        return
    csv_filepath = get_file_path('users.csv')
    fields = ['id', 'name', 'email', 'phone', 'age', 'gender', 'location', 'traits', 'personality_type', 'created_at']
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for user in users:
            row = {}
            for k in fields:
                if k == 'traits':
                    row[k] = ",".join(user.get('traits', []))
                else:
                    row[k] = user.get(k, '')
            writer.writerow(row)
