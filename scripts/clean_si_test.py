from pathlib import Path
from ndi_robot_registration.clean_si_data import clean_si_data
from ndi_robot_registration.clean_ndi_data import load_ndi_data, parse_all_tools

# input_csv = Path("data/data_local_part_1_18_7_2026_20_49_41.csv")
# cleaned_data = clean_si_data(input_csv, timestamp_column="Time_stamp", arm_column="mip Index (PSM1)")
# cleaned_data.to_csv("data/cleaned_data_local_part_1_18_7_2026_20_49_41.csv", index=False)

input_csv = Path("data/20260718_c.csv")
ndi_data = load_ndi_data(input_csv)
tools_data = parse_all_tools(ndi_data)
key = 1
if key in tools_data:
    print(tools_data[key].head())  # Print the first few rows of the data for Tool ID 1
else:
    print(f"Tool ID {key} not found in data. Available Tools: {list(tools_data.keys())}")