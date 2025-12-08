import numpy as np
np.set_printoptions(threshold=10000)
from utils.data_loader import prepare_data, load_market_price, load_fr_signal

fr_files="real_data/FR signal_data/04 2024.xlsx"
price_files="real_data/price_data/04 2024.xlsx"
(prices, fr_signals) = prepare_data(fr_files, price_files, mode="train", point="HB_NORTH", fr_start_date="2024/4/7", price_start_date="2024-04-07")

print("处理后的电价数据:", prices)
print("电价数据长度:", len(prices))
print("处理后的FR信号数据:", fr_signals)
print("FR信号数据长度:", len(fr_signals))