import os
import sys
import random
from datetime import datetime, timedelta

# Ensure python path includes the backend directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_handler import load_json, save_json, sync_users_to_csv, convert_list_to_dict
from utils.initializer import initialize_user_files

FIRST_NAMES_MALE = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Rayaan", "Ayaan", "Krishna", "Ishaan", "Rohan", "Rahul", "Amit", "Vikram", "Siddharth"]
FIRST_NAMES_FEMALE = ["Ananya", "Diya", "Aadhya", "Saanvi", "Avni", "Kiara", "Isha", "Riya", "Myra", "Aarohi", "Priya", "Neha", "Sneha", "Kavya", "Pooja"]
LAST_NAMES = ["Sharma", "Patel", "Singh", "Kumar", "Gupta", "Deshmukh", "Joshi", "Verma", "Reddy", "Rao", "Iyer", "Nair", "Das", "Bose", "Mehta"]
LOCATIONS = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat", "Lucknow", "Kanpur"]

TRAITS_POOL = [
    "analytical", "creative", "introverted", "extroverted", "leader", 
    "empathetic", "practical", "observant", "logical", "organized", 
    "spontaneous", "adaptable", "determined", "ambitious", "curious", 
    "resilient", "diplomatic", "adventurous", "methodical"
]

MBTI_TYPES = [
    "INTJ", "INFP", "ENFP", "ENTJ", "ISTP", "ISFP", "ESTP", "ESFP", 
    "INFJ", "ENFJ", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "INTP", "ENTP"
]

def random_date(start, end):
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())),
    )

def generate_dummy_users_with_traits(count=60):
    users_data = load_json('users.json', default_value={})
    users = convert_list_to_dict(users_data, key='email')
        
    start_id = 1
    if users:
        start_id = max(u.get('id', 0) for u in users.values()) + 1

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365) # Generate dates over the last year

    emails_generated = set(users.keys())

    added_count = 0
    while added_count < count:
        gender = random.choice(["Male", "Female"])
        if gender == "Male":
            first_name = random.choice(FIRST_NAMES_MALE)
        else:
            first_name = random.choice(FIRST_NAMES_FEMALE)
            
        last_name = random.choice(LAST_NAMES)
        name = f"{first_name} {last_name}"
        
        email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 9999)}@example.com"
        
        if email in emails_generated:
            continue
            
        emails_generated.add(email)
        
        phone = f"+91 {random.randint(6, 9)}{random.randint(100000000, 999999999)}"
        age = random.randint(18, 50)
        location = random.choice(LOCATIONS)
        
        # Traits and Personality
        num_traits = random.randint(3, 6)
        user_traits = random.sample(TRAITS_POOL, num_traits)
        personality_type = random.choice(MBTI_TYPES)
        
        created_at = random_date(start_date, end_date).isoformat() + "Z"

        user = {
            "id": start_id,
            "name": name,
            "email": email,
            "phone": phone,
            "age": age,
            "gender": gender,
            "location": location,
            "traits": user_traits,
            "personality_type": personality_type,
            "created_at": created_at
        }
        
        users[email] = user
        
        # Initialize other files for this user
        initialize_user_files(email)
        
        start_id += 1
        added_count += 1
        
    save_json('users.json', users)
    sync_users_to_csv()
    
    print(f"Successfully generated {count} dummy users with traits and initialized their data in all related JSON files.")

if __name__ == "__main__":
    generate_dummy_users_with_traits()
