import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Setup Gemini
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Error: GOOGLE_API_KEY not found in environment.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

def optimize_query(user_input: str) -> str:
    """
    Rewrites user input to match the specific text serialization format 
    of the airline vector database.
    """
    print(f"   [Tool] Optimizing query: '{user_input}'")
    
    system_instruction = """
    You are a query optimizer for an airline database. 
    Rewrite the user's input to match our specific text serialization format.

    ### TRANSLATION RULES
    * Good Food -> "The food was delicious and excellent"
    * Bad Food  -> "The food was terrible and poor"
    * Average   -> "The food was average"
    * Late      -> "The flight was significantly delayed"
    * On Time   -> "The flight was on time"
    """
    
    prompt = f"{system_instruction}\n\nUser: {user_input}\nOptimized:"
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"   [Error] Prompt Engineer failed: {e}")
        return user_input # Fallback