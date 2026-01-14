from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from swvo.io.symh import SymhOMNI
import matplotlib.pyplot as plt
from datetime import timedelta, timezone

event_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

symh_data = SymhOMNI(".swvo").read(
    event_time - timedelta(days=10),
    event_time + timedelta(days=10),
    download=True,
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(symh_data.index, symh_data["sym-h"], label="SYM-H", color="tab:blue")


ax.set_xlabel("Time (UTC)")
ax.set_ylabel("SYM-H (nT)")
ax.set_title("OMNI SYM-H around 22 Apr 2017 23:58 UTC")
ax.legend()

plt.tight_layout()
plt.show()



# # from swvo.io.symh import SymhOMNI

# from datetime import datetime, timezone
# import os

# print("="*70)
# print("Testing SymhOMNI: January 2012 to December 2017")
# print("="*70)

# # Initialize
# data_dir = '/home/parvathy/SWVO/test_data'
# symh = SymhOMNI(data_dir=data_dir)

# # Define time range: Jan 1, 2012 to Dec 31, 2017
# start = datetime(2012, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
# end = datetime(2017, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# print(f"\nTime Range:")
# print(f"  Start: {start}")
# print(f"  End:   {end}")
# print(f"  Duration: 6 years")


# cadence = 5 

# print(f"\nCadence: {cadence} minutes")
# print("Downloading and processing data...")
# print("This may take several minutes for 6 years of data...\n")

# # Read data
# df = symh.read(start, end, cadence_min=cadence, download=True)

# # Calculate expected rows
# total_days = (end - start).days + 1
# expected_rows = total_days * 24 * (60 // cadence)

# print("\n" + "="*70)
# print("DATA SUMMARY")
# print("="*70)

# print(f"\nTotal rows: {len(df):,}")
# print(f"Expected rows (~{cadence}-min cadence): ~{expected_rows:,}")
# print(f"Columns: {df.columns.tolist()}")

# print(f"\nDate range in data:")
# print(f"  First timestamp: {df.index[0]}")
# print(f"  Last timestamp:  {df.index[-1]}")

# print(f"\nSYM-H Statistics:")
# print(df['sym-h'].describe())

# print(f"\nMissing data:")
# print(f"  NaN count: {df['sym-h'].isna().sum():,}")
# print(f"  Valid data: {df['sym-h'].notna().sum():,}")
# print(f"  Coverage: {(df['sym-h'].notna().sum() / len(df) * 100):.2f}%")

# # Save to CSV
# output_path = os.path.join(data_dir, f'symh_{cadence}min_2012_2017.csv')
# print(f"\nSaving to CSV...")
# df.to_csv(output_path)

# print("\n" + "="*70)
# print(f"✓ CSV SAVED TO: {output_path}")
# print("="*70)

# # Show file size
# file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
# print(f"File size: {file_size_mb:.2f} MB")

# print(f"\nFIRST 10 ROWS:")
# print(df.head(10))

# print(f"\nLAST 10 ROWS:")
# print(df.tail(10))

# # Show some interesting statistics
# print(f"\nINTERESTING STATISTICS:")
# print(f"  Minimum SYM-H: {df['sym-h'].min():.1f} nT")
# print(f"  Maximum SYM-H: {df['sym-h'].max():.1f} nT")
# print(f"  Mean SYM-H: {df['sym-h'].mean():.1f} nT")
# print(f"  Median SYM-H: {df['sym-h'].median():.1f} nT")

# # Find extreme events
# print(f"\nEXTREME EVENTS (Storm conditions, SYM-H < -100):")
# storm_events = df[df['sym-h'] < -100]
# if len(storm_events) > 0:
#     print(f"  Found {len(storm_events):,} data points with SYM-H < -100 nT")
#     print(f"  Strongest storm: {storm_events['sym-h'].min():.1f} nT at {storm_events['sym-h'].idxmin()}")
# else:
#     print("  No major storms found (SYM-H >= -100)")

# print("\n" + "="*70)
# print("✓ Test completed successfully!")
# print("="*70)














# from swvo.io.omni import OMNIHighRes
# from datetime import datetime, timezone
# import os

# # Initialize with your test_data directory
# data_dir = '/home/parvathy/SWVO/test_data'
# omni_hr = OMNIHighRes(data_dir=data_dir)

# # Get one day of data (1-minute cadence)
# start = datetime(2012, 1, 1, tzinfo=timezone.utc)
# end = datetime(2017, 12, 31, tzinfo=timezone.utc)

# print("Downloading and reading OMNI high-resolution data...")
# df = omni_hr.read(start, end, cadence_min=1, download=True)

# # Save to CSV in the test_data directory
# output_path = os.path.join(data_dir, 'omni_highres_output.csv')
# df.to_csv(output_path)

# print("\n" + "="*70)
# print(f"✓ CSV SAVED TO: {output_path}")
# print("="*70)

# print(f"\nTotal rows: {len(df)}")
# print(f"\nHEADER (Columns):")
# print(df.columns.tolist())

# print(f"\nFIRST 10 ROWS:")
# print(df.head(10))

# print("\n" + "="*70)
# print(f"Data range: {df.index[0]} to {df.index[-1]}")
# print("="*70)