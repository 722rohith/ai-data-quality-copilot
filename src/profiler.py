import pandas as pd

def profile_dataset(file_path_or_buffer):
    df = pd.read_csv(file_path_or_buffer)
    
    # 🛠️ Automatically drop completely empty columns (like trailing 'Unnamed' columns)
    df = df.dropna(how='all', axis=1)
    
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