import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)
model = genai.GenerativeModel("gemini-2.5-flash")
def generate(prompt):
    """
    Generate an AI response using Gemini.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"""
## Gemini Error
{str(e)}
"""