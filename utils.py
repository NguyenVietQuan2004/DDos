import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import socket
import struct

def ip_to_int(ip):
    """Chuyển địa chỉ IP dạng chuỗi sang số nguyên 32-bit."""
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except:
        return 0  # Fallback nếu IP không hợp lệ

def preprocess_input_data(df, scaler, n_past=30):
    """
    Tiền xử lý dữ liệu đầu vào cho mô hình LSTM.
    
    Parameters:
    - df: DataFrame từ file CSV đầu vào
    - scaler: StandardScaler đã được fit từ dữ liệu huấn luyện
    - n_past: Số bước thời gian trong chuỗi (mặc định 30, khớp với trainX_30_timeseries.pkl)
    
    Returns:
    - X: Dữ liệu đã tiền xử lý, shape (n_samples, n_past, n_features)
    """
    # In danh sách cột để debug
    # print("Columns in DataFrame before processing:", df.columns.tolist())

    # Kiểm tra cột ' Label'
    if ' Label' not in df.columns:
        raise KeyError(f"Cột ' Label' không tồn tại. Các cột hiện có: {df.columns.tolist()}")

    # 1. Chuẩn hóa và sắp xếp thời gian
    df[' Timestamp'] = pd.to_datetime(df[' Timestamp'], errors='coerce')
    df = df.dropna(subset=[' Timestamp'])  # Xóa dòng có Timestamp không hợp lệ
    df.sort_values(by=' Timestamp', inplace=True)
    # print("Values in ' Label' column before encoding:", df[' Label'].unique())

    # 2. Xóa các cột không cần thiết
    constant_columns = [
        ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' CWE Flag Count',
        'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate',
        ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
    ]
    df.drop(['Flow ID', ' Fwd Header Length.1'], axis=1, inplace=True, errors='ignore')
    df.drop(columns=constant_columns, inplace=True, errors='ignore')
    # print("Columns after dropping:", df.columns.tolist())

    # 3. Mã hóa nhãn và địa chỉ IP
    label_encoder = LabelEncoder()
    df[' Label'] = label_encoder.fit_transform(df[' Label'])
    df[' Source IP'] = df[' Source IP'].apply(ip_to_int)
    df[' Destination IP'] = df[' Destination IP'].apply(ip_to_int)

    # 4. Đặt Timestamp làm index
    df.set_index(' Timestamp', inplace=True)

    # 5. Lấy danh sách cột không nhị phân từ scaler
    non_binary_columns = [col for col in scaler.feature_names_in_ if col in df.columns]

    # 6. Kiểm tra các cột mà scaler mong đợi
    scaler_columns = scaler.feature_names_in_
    # print("Scaler expected columns:", scaler_columns.tolist())
    missing_columns = [col for col in scaler_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Các cột thiếu trong DataFrame: {missing_columns}")

    # 7. Ép kiểu dữ liệu thành số cho các cột không nhị phân
    for col in non_binary_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 8. Loại bỏ các dòng chứa giá trị vô cực
    rows_with_inf = df[non_binary_columns].apply(lambda x: np.isinf(x)).any(axis=1)
    df = df[~rows_with_inf].copy()

    # 9. Chuẩn hóa các cột không nhị phân
    df[non_binary_columns] = scaler.transform(df[non_binary_columns])

    # 10. Tạo chuỗi thời gian cho LSTM
    X = []
    for i in range(n_past, len(df)):
        X.append(df.iloc[i - n_past:i, :].values)  # Lấy chuỗi n_past bước
    X = np.array(X)  # Shape: (n_samples, n_past, n_features)

    return X