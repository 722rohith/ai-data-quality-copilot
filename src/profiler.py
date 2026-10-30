import pandas as pd
import os

def profile_dataset(file_path_or_buffer):
    # Get file name safely from buffer or string path
    file_name = "dataset"
    if hasattr(file_path_or_buffer, 'name'):
        file_name = os.path.splitext(os.path.basename(file_path_or_buffer.name))[0]
    elif isinstance(file_path_or_buffer, str):
        file_name = os.path.splitext(os.path.basename(file_path_or_buffer))[0]

    df = pd.read_csv(file_path_or_buffer)
    
    # Automatically rename 'Unnamed' columns using the file name prefix
    df.columns = [
        f"{file_name}_col_{i}" if str(col).startswith('Unnamed') else col 
        for i, col in enumerate(df.columns)
    ]
    
    total_rows = len(df)
    total_columns = len(df.columns)
    missing_values = df.isnull().sum().to_dict()
    missing_percentage = (df.isnull().sum() / total_rows * 100).to_dict()
    duplicate_count = df.duplicated().sum()
    dtypes = df.dtypes.astype(str).to_dict()

    profile_summary = {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "duplicate_rows": int(duplicate_count),
        "missing_values": missing_values,
        "missing_percentage": missing_percentage,
        "data_types": dtypes
    }
    return df, profile_summary