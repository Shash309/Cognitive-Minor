import os
import re

base_dir = r'c:\Users\HP\OneDrive\Desktop\Cognitive-Minor'
input_file = os.path.join(base_dir, 'important_system_codes.md')

def clean_code(code_str):
    lines = code_str.split('\n')
    clean_lines = []
    for line in lines:
        if line.startswith('# %%') or line.startswith('pip install') or line.strip() == 'c':
            continue
        
        if '#' in line:
            if '"' not in line and "'" not in line:
                line = line.split('#')[0]
            elif line.strip().startswith('#'):
                continue
        
        line = line.rstrip()
        if line:  
            clean_lines.append(line)
    return '\n'.join(clean_lines)

intel_engine = ''
quiz_engine = ''
model_training = ''

with open(os.path.join(base_dir, 'backend', 'career_intelligence_engine.py'), 'r', encoding='utf-8') as f:
    intel_engine = clean_code(f.read())

with open(os.path.join(base_dir, 'backend', 'career_quiz_engine.py'), 'r', encoding='utf-8') as f:
    quiz_engine = clean_code(f.read())

notebook_path = os.path.join(base_dir, 'backend', '# %% [markdown].py')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_content = f.read()
    
match = re.search(r'# STEP 1: Imports(.*)', nb_content, re.DOTALL)
if match:
    final_model_code = "import pandas as pd\n" + match.group(1) 
    model_training = clean_code(final_model_code)
else:
    model_training = clean_code(nb_content)

with open(input_file, 'w', encoding='utf-8') as f:
    f.write('# Important System Codes (Cleaned & Minimized)\n\n')
    f.write('## Career Intelligence Engine (Fusion Logic)\n\n')
    f.write('```python\n')
    f.write(intel_engine)
    f.write('\n```\n\n')
    
    f.write('## Career Quiz Engine (Adaptive Logic)\n\n')
    f.write('```python\n')
    f.write(quiz_engine)
    f.write('\n```\n\n')
    
    f.write('## Model Training Script (Final Ensemble Pipeline)\n\n')
    f.write('```python\n')
    f.write(model_training)
    f.write('\n```\n\n')

print('Cleaned artifact successfully.')
