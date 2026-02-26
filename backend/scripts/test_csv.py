
import pandas as pd
import os
import numpy as np

try:
    df = pd.read_csv("colleges.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    print("Columns:", df.columns.tolist())
    
    state = "Delhi"
    state_colleges = df[df["state"].str.lower() == state.lower()]
    print(f"Colleges in {state}: {len(state_colleges)}")
    
    if len(state_colleges) > 0:
        agg_dict = {
            'city': ('city', 'first'),
            'rankings': ('ranking', list),
            'fields': ('field', list),
            'scores': ('score', list)
        }
        if 'website' in state_colleges.columns:
             agg_dict['website'] = ('website', 'first')

        print("Agg dict:", agg_dict)
        grouped = state_colleges.groupby('college_name').agg(**agg_dict).reset_index()
        print("Grouped head:")
        print(grouped.head())
        
        def get_best_ranking(rankings):
            valid = [r for r in rankings if pd.notnull(r)]
            return min(valid) if valid else np.inf

        grouped['best_ranking'] = grouped['rankings'].apply(get_best_ranking)
        print("Best ranking calculated.")
        
except Exception as e:
    print("Error:", e)
