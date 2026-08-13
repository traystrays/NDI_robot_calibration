from pathlib import Path
import pandas as pd
from ndi_robot_registration.clean_si_data import clean_si_data
from ndi_robot_registration.clean_ndi_data import clean_ndi_data
from ndi_robot_registration.match import match


pd.set_option("display.width", None)

input_csv = Path("data/data_local_part_1_18_7_2026_20_49_41.csv")
cleaned_si_data = clean_si_data(input_csv, timestamp_column="Time_stamp", arm_column="mip Index (PSM1)")
cleaned_si_data.to_csv("data/cleaned_data_local_part_1_18_7_2026_20_49_41.csv", index=False)
print(cleaned_si_data.head())  # Print the first few rows of the cleaned data

input_csv_ndi = Path("data/20260718_a.csv")
cleaned_ndi_data = clean_ndi_data(input_csv_ndi, timestamp_column="Timestamp", toolkey=1)
print(cleaned_ndi_data.head())  # Print the first few rows of the cleaned NDI data

matched_data, original_match, current_length, dropped_count = match(cleaned_si_data, cleaned_ndi_data, tolerance="10ms")
print(f"Matched data length: {current_length}, Dropped count: {dropped_count}")
print(original_match.head())  # Print the first few rows of the original matched data


matched_data.columns = ["timestamp", "SI Transform", "NDI Transform"]

print(matched_data.head())  # Print the first few rows of the matched data

matched_data.to_csv("data/matched_data_local_part_1_18_7_2026_20_49_41.csv", index=False)
original_match.to_csv("data/original_matched_data_local_part_1_18_7_2026_20_49_41.csv", index=False)