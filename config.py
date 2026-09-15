from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF")

def get_var():
    return hf_token