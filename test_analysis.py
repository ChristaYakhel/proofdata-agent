import pandas as pd
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load real data
employees = pd.read_csv("data/employees.csv")

# User question
question = "What is the average salary of employees?"

# Ask Gemini for code
prompt = f"""
You are a data analyst.

A pandas DataFrame named 'employees' already contains the user's data.

Columns:
{list(employees.columns)}

User question:
{question}

Generate ONLY Python code to answer the question.

Rules:
- Use the existing DataFrame named employees.
- Do not create a new DataFrame.
- Do not create sample data.
- Do not use fake values.
- Use only the columns provided.
- Store the final answer in a variable called result.
- Return only Python code.
"""

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)

code = response.text

# Remove markdown
code = code.replace("```python", "")
code = code.replace("```", "")

print("========== GENERATED CODE ==========")
print(code)

# Execute Gemini code
local_variables = {
    "employees": employees
}

exec(code, {}, local_variables)

ai_result = local_variables.get("result")

print("\n========== AI RESULT ==========")
print(ai_result)

# Independent verification
verified_result = employees["salary"].mean()

print("\n========== INDEPENDENT RESULT ==========")
print(verified_result)

# Compare results
if ai_result == verified_result:
    print("\n========== VERIFICATION ==========")
    print("✅ VERIFIED")
    print("AI result matches independent calculation.")
else:
    print("\n========== VERIFICATION ==========")
    print("❌ VERIFICATION FAILED")
    print("AI result does not match the independent calculation.")

