import os
import sys
import io
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import functions from ashare_data_fetcher
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from ashare_data_fetcher import fetch_and_save_stock

if __name__ == "__main__":
    # Postal Savings Bank of China (邮储银行)
    symbol = "601658"
    name = "邮储银行"
    fetch_and_save_stock(symbol, name, force=True)
