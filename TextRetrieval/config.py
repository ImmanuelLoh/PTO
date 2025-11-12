import glob 

COMPANY_NAME = "Google"

DATA_DIR = "00-data"
CHUNK_SIZE = 500  # number of words per chunk 


# Annual reports (10-Ks)
annual_files = glob.glob(f"{DATA_DIR}/annuals/*.pdf")
# # Quarterly reports (10-Qs)
quarterly_files = glob.glob(f"{DATA_DIR}/quarterlies/*.pdf")