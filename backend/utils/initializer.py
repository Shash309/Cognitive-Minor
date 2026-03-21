from utils.data_handler import load_json, save_json, convert_list_to_dict

def dict_add_entry(filename, email, entry_data):
    data = load_json(filename, default_value={})
    data = convert_list_to_dict(data, key="email")
    data[email] = entry_data
    save_json(filename, data)

def initialize_user_files(email):
    # psych_profiles.json
    dict_add_entry('psych_profiles.json', email, [])

    # quiz_history.json
    dict_add_entry('quiz_history.json', email, {
        "attempts": []
    })

    # voice_history.json
    dict_add_entry('voice_history.json', email, [])

    # user_progress.json
    dict_add_entry('user_progress.json', email, {
        "psych_completed": False,
        "voice_completed": False,
        "quiz_completed": False,
        "last_completed_step": None
    })

    # career_sessions.json
    dict_add_entry('career_sessions.json', email, [])

    # career_fused_results.json
    dict_add_entry('career_fused_results.json', email, {
        "result": None
    })
