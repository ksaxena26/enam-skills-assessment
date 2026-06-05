import io
import os
import msoffcrypto
from src.config.paths import OHLCV_DIR, TB_DIR
from src.utils.secrets import tb_pass
import pandas as pd


def decrypt_tradebook():
    # Define your file path and the required password
    tb_file_path = TB_DIR / "hist_trade_data.xlsx"
    # password = "your_password_here"

    # Create an in-memory byte stream and decrypt file
    decrypted_data = io.BytesIO()
    with open(tb_file_path, "rb") as f:
        office_file = msoffcrypto.OfficeFile(f)
        office_file.load_key(password=tb_pass)
        office_file.decrypt(decrypted_data)

    # Read directly into Pandas
    decrypted_data.seek(0)
    return decrypted_data


def load_ohlc_data() -> pd.DataFrame:
    ohlc_df = pd.DataFrame()
    for file in os.listdir(OHLCV_DIR):
        ohlc_df = pd.concat([ohlc_df, pd.read_csv(OHLCV_DIR / file)], ignore_index=True)

    ohlc_df['date'] = pd.to_datetime(ohlc_df['date'], format='%Y-%m-%d')
    ohlc_df = ohlc_df.drop_duplicates(['date', 'symbol'])
    ohlc_df = ohlc_df.sort_values(['symbol', 'date'], ascending=[True, True]).reset_index(drop=True)

    return ohlc_df

if __name__ == "__main__":
    # data_d = decrypt_tradebook()
    # tb_df = pd.read_excel(data_d)
    price_df = load_ohlc_data()
    print(price_df.head())

    print("End of read")