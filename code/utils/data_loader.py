import pandas as pd
import numpy as np
import os

def load_fr_signal(fr_paths, max_points=None, start_date=None) -> np.ndarray:
    """
    支持单文件(str)或多文件(list)的FR信号加载与拼接
    """
    if isinstance(fr_paths, str):
        fr_paths = [fr_paths]
    arrays = []
    for fp in fr_paths:
        if not os.path.exists(fp):
            print(f"[警告] 未找到FR signals文件: {fp}")
            continue
        print(f"[信息] 正在加载FR signals文件: {fp}")
        df = pd.read_excel(fp, sheet_name="Dynamic", header=None)
        if start_date:
            # 提取第一行日期戳（跳过第一列）
            date_stamps = df.iloc[0, 1:].values
            
            # 日期格式兼容处理
            date_stamps = pd.to_datetime(date_stamps, errors='coerce')
            start_date_dt = pd.to_datetime(start_date, errors='coerce')
            # 找到起始日期位置
            start_idx = np.where(date_stamps >= start_date_dt)[0]
            if len(start_idx) > 0:
                start_col = start_idx[0] # 列索引偏移
                df = df.iloc[:, start_col:]   # 保留起始日期后的列
            else:
                print(f"[警告] 未找到{start_date}后的数据")
        df = df.drop(columns=df.columns[0])
        df = df.iloc[1:, :] 
        arr = df.values.flatten(order='F').astype(np.float32)
        arrays.append(arr)
    if not arrays:
        raise FileNotFoundError("未找到任何FR signals文件,请检查路径!")
    fr_all = np.concatenate(arrays)
    if max_points is not None:
        if len(fr_all) < max_points:
            print(f"[警告] FR signals总数据量不足: 需要{max_points}，实际{len(fr_all)}")
        fr_all = fr_all[:max_points]
    return fr_all

def load_market_price(price_paths, point_name="HB_NORTH", max_points=None, start_date=None) -> np.ndarray:
    """
    支持单文件(str)或多文件(list)的电价加载与拼接
    """
    if isinstance(price_paths, str):
        price_paths = [price_paths]
    arrays = []
    for fp in price_paths:
        if not os.path.exists(fp):
            print(f"[警告] 未找到market prices文件: {fp}")
            continue
        print(f"[信息] 正在加载market prices文件: {fp}")
        df = pd.read_excel(fp, sheet_name='Normal')
        df = df[df["Settlement Point Name"] == point_name]
        df["Timestamp"] = pd.to_datetime(df["Delivery Date"]) + pd.to_timedelta(df["Delivery Hour"], unit="h") + pd.to_timedelta(df["Delivery Interval"]*15, unit="m")
        if start_date:
            df = df[df["Timestamp"] >= start_date]
        df = df.sort_values("Timestamp")
        arr = df["Settlement Point Price"].values.astype(np.float32)
        arrays.append(arr)
    if not arrays:
        raise FileNotFoundError("未找到任何电价文件,请检查路径!")
    price_all = np.concatenate(arrays)
    if max_points is not None:
        if len(price_all) < max_points:
            print(f"[警告] 电价总数据量不足: 需要{max_points}，实际{len(price_all)}")
        price_all = price_all[:max_points]
    return price_all

def downsample_fr(fr_array, factor = 5) -> np.ndarray:
    """
    将 2 秒采样的 FR 信号按平均值每 10 秒降采样（训练用）
    """
    length = len(fr_array)
    trimmed = fr_array[:length - (length % factor)]  # 裁剪多余部分
    return trimmed.reshape(-1, factor).mean(axis=1)


def prepare_data(fr_path, price_path, mode="train", point="HB_NORTH", fr_start_date=None, price_start_date=None) -> tuple:
    if mode=="train":
        days = 7
    elif mode=="test":
        days = 60

    fr_max_points = 43200 * days if days is not None else None
    price_max_points = 96 * days if days is not None else None

    fr_raw = load_fr_signal(fr_path, max_points=fr_max_points, start_date=fr_start_date)
    price_raw = load_market_price(price_path, point_name=point, max_points=price_max_points, start_date=price_start_date)

    if mode == "train":
        fr_processed = downsample_fr(fr_raw, factor=5)
        time_step = 10
    elif mode == "test":
        fr_processed = fr_raw
        time_step = 2
    else:
        raise ValueError("Mode must be 'train' or 'test'")
    
    # 电价插值到与FR信号相同的时间步长
    price_interval = 15 * 60  # 15分钟=900秒
    repeat_num = price_interval // time_step

    # 扩展电价序列
    expanded_price = np.repeat(price_raw, repeat_num)
    min_len = min(len(expanded_price), len(fr_processed))
    return expanded_price[:min_len], fr_processed[:min_len]