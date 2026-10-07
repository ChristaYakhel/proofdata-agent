import streamlit as st
import pandas as pd
import numpy as np
import math
import os
import re
import ast

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. SETUP
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY is missing in .env")
    st.stop()

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.5-flash-lite"


# ============================================================
# 2. PAGE
# ============================================================

st.set_page_config(
    page_title="ProofData AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 ProofData AI")
st.caption(
    "A verifiable AI data analyst — every answer comes with executable proof."
)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "question" not in st.session_state:
    st.session_state.question = ""

if "plan" not in st.session_state:
    st.session_state.plan = ""

if "code" not in st.session_state:
    st.session_state.code = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "verified" not in st.session_state:
    st.session_state.verified = False

if "run_count" not in st.session_state:
    st.session_state.run_count = 0

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "cannot_answer" not in st.session_state:
    st.session_state.cannot_answer = False


# ============================================================
# 4. DATAFRAME NAME
# ============================================================

def safe_dataframe_name(filename):

    name = os.path.splitext(filename)[0]

    name = re.sub(
        r"[^a-zA-Z0-9_]",
        "_",
        name
    )

    if name and name[0].isdigit():
        name = "table_" + name

    return name.lower()


# ============================================================
# 5. DATA SUMMARY
# ============================================================

def create_data_summary(dataframes):

    summary = ""

    for name, df in dataframes.items():

        summary += f"\nTABLE: {name}\n"

        summary += f"Rows: {len(df)}\n"

        summary += f"Columns: {list(df.columns)}\n"

        summary += "Column types:\n"

        for column in df.columns:

            summary += (
                f"  {column}: {df[column].dtype}\n"
            )

        summary += (
            f"Missing values: "
            f"{int(df.isnull().sum().sum())}\n"
        )

        summary += (
            f"Duplicate rows: "
            f"{int(df.duplicated().sum())}\n"
        )

        summary += "\nSample data:\n"

        summary += df.head(5).to_string(
            index=False
        )

        summary += "\n\n"

    # Find relationships

    names = list(dataframes.keys())

    if len(names) > 1:

        summary += "TABLE RELATIONSHIPS:\n"

        for i in range(len(names)):

            for j in range(i + 1, len(names)):

                df1 = dataframes[names[i]]
                df2 = dataframes[names[j]]

                shared = []

                for column in df1.columns:

                    if column in df2.columns:

                        shared.append(column)

                if shared:

                    summary += (
                        f"{names[i]} <-> "
                        f"{names[j]} : "
                        f"{shared}\n"
                    )

    return summary


# ============================================================
# 6. DATA QUALITY
# ============================================================

def get_warnings(dataframes):

    warnings = []

    for name, df in dataframes.items():

        # Missing values

        missing = df.isnull().sum()

        for column, count in missing.items():

            if count > 0:

                warnings.append(
                    f"{name}.{column} has "
                    f"{count} missing value(s)."
                )

        # Duplicate rows

        duplicate_count = int(
            df.duplicated().sum()
        )

        if duplicate_count > 0:

            warnings.append(
                f"{name} contains "
                f"{duplicate_count} duplicate row(s)."
            )

        # Currency

        for column in df.columns:

            if "currency" in column.lower():

                values = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if len(values) > 1:

                    warnings.append(
                        f"{name}.{column} contains "
                        f"multiple currencies: {values}"
                    )

        # Ambiguous dates

        for column in df.columns:

            if "date" in column.lower():

                values = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .head(20)
                )

                for value in values:

                    if re.match(
                        r"^\d{1,2}/\d{1,2}/\d{4}$",
                        value
                    ):

                        warnings.append(
                            f"{name}.{column} may contain "
                            f"ambiguous date: {value}"
                        )

                        break

    # Remove duplicates

    return list(dict.fromkeys(warnings))


# ============================================================
# 7. AI PLANNER
# ============================================================

def create_plan(question, summary):

    prompt = f"""
You are the reasoning engine of ProofData AI.

The user wants an answer from uploaded datasets.

Your job is to create a precise plan.

VERY IMPORTANT:

- Understand exactly what the USER asks.
- Never answer a different question.
- A shared column is NOT the answer unless the user asks
  for shared columns.
- Shared columns are normally used only for joining tables.
- Never invent a column.
- Never invent data.
- If the available data cannot reliably answer the question,
  return CANNOT_ANSWER.
- If missing data prevents the answer, return CANNOT_ANSWER.
- If contradictory records prevent a reliable answer,
  return CANNOT_ANSWER.
- If units/currencies cannot safely be compared,
  return CANNOT_ANSWER.
- If an ambiguous date prevents the requested calculation,
  return CANNOT_ANSWER.

USER QUESTION:
{question}

AVAILABLE DATA:
{summary}

Return ONLY:

DECISION: ANSWER or CANNOT_ANSWER

REASON: short explanation

TARGET: exactly what the user wants

TABLES: required tables

COLUMNS: required columns

OPERATION: exact calculation/filter/sort/group operation

JOIN: exact join condition or NONE

WARNINGS: important problems or NONE
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# 8. AI CODE GENERATOR
# ============================================================

def generate_code(question, plan, summary):

    prompt = f"""
You generate executable Python proof for ProofData AI.

USER QUESTION:
{question}

REASONING PLAN:
{plan}

AVAILABLE DATA:
{summary}

RULES:

1. Use ONLY the uploaded pandas DataFrames.
2. Do not create fake data.
3. Do not import anything.
4. Do not use external files.
5. Do not use internet.
6. Do not use:
   os
   sys
   subprocess
   requests
   urllib
   socket
   pathlib
   open
   eval
   exec
7. Use pandas operations.
8. If multiple tables are required, perform the correct merge.
9. Do exactly what the question asks.
10. Do not answer a different question.
11. The final answer MUST be stored in `result`.
12. Do not use print().
13. Return ONLY Python code.

Example:

merged = employees.merge(
    sales,
    on="employee_id",
    how="inner"
)

result = merged.loc[
    merged["sales"].idxmax(),
    "name"
]

Return ONLY the Python code.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return clean_code(response.text)


# ============================================================
# 9. CLEAN CODE
# ============================================================

def clean_code(code):

    code = code.strip()

    if "```" in code:

        code = re.sub(
            r"```python",
            "",
            code,
            flags=re.IGNORECASE
        )

        code = code.replace(
            "```",
            ""
        )

    return code.strip()


# ============================================================
# 10. CODE VALIDATION
# ============================================================

def validate_code(code):

    dangerous = {
        "os",
        "sys",
        "subprocess",
        "requests",
        "urllib",
        "socket",
        "pathlib",
        "open",
        "eval",
        "exec",
        "compile",
        "__import__"
    }

    try:

        tree = ast.parse(code)

    except SyntaxError as e:

        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):

        # Block imports

        if isinstance(
            node,
            (ast.Import, ast.ImportFrom)
        ):

            return False, "Imports are not allowed."

        # Block dangerous functions

        if isinstance(node, ast.Name):

            if node.id in dangerous:

                return False, (
                    f"Dangerous operation: {node.id}"
                )

        # Block private attributes

        if isinstance(node, ast.Attribute):

            if node.attr.startswith("__"):

                return False, (
                    "Private attributes are not allowed."
                )

    return True, "Code is safe to execute."


# ============================================================
# 11. EXECUTE PROOF
# ============================================================

def execute_proof(code, dataframes):

    valid, message = validate_code(code)

    if not valid:

        return False, None, message

    # Fresh copies every time
    local_dataframes = {}

    for name, df in dataframes.items():

        local_dataframes[name] = df.copy()

    # Safe environment

    environment = {

        "pd": pd,
        "np": np,
        "math": math,

        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "round": round,
        "abs": abs,

        "float": float,
        "int": int,
        "str": str,

        "list": list,
        "set": set,
        "dict": dict,
        "tuple": tuple,

        "range": range,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,

        "any": any,
        "all": all
    }

    environment.update(
        local_dataframes
    )

    try:

        exec(
            code,
            {"__builtins__": {}},
            environment
        )

    except Exception as e:

        return False, None, (
            f"Execution error: {e}"
        )

    if "result" not in environment:

        return False, None, (
            "Proof code did not create `result`."
        )

    result = environment["result"]

    return True, result, "Success"


# ============================================================
# 12. COMPARE RESULTS
# ============================================================

def compare_results(a, b):

    try:

        if isinstance(a, pd.DataFrame):

            return (
                isinstance(b, pd.DataFrame)
                and a.equals(b)
            )

        if isinstance(a, pd.Series):

            return (
                isinstance(b, pd.Series)
                and a.equals(b)
            )

        if isinstance(a, np.ndarray):

            return (
                isinstance(b, np.ndarray)
                and np.array_equal(
                    a,
                    b,
                    equal_nan=True
                )
            )

        return a == b

    except Exception:

        return str(a) == str(b)


# ============================================================
# 13. DISPLAY RESULT
# ============================================================

def show_result(result):

    if isinstance(result, pd.DataFrame):

        st.dataframe(
            result,
            use_container_width=True
        )

    elif isinstance(result, pd.Series):

        st.dataframe(
            result,
            use_container_width=True
        )

    elif isinstance(result, np.ndarray):

        st.write(
            result.tolist()
        )

    else:

        st.markdown(
            f"### {result}"
        )


# ============================================================
# 14. UPLOAD DATA
# ============================================================

st.sidebar.header("📁 Upload Data")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)

dataframes = {}

if uploaded_files:

    for file in uploaded_files:

        try:

            df = pd.read_csv(file)

            name = safe_dataframe_name(
                file.name
            )

            original = name
            counter = 2

            while name in dataframes:

                name = (
                    f"{original}_{counter}"
                )

                counter += 1

            dataframes[name] = df

        except Exception as e:

            st.error(
                f"Could not read {file.name}: {e}"
            )


# ============================================================
# 15. SHOW DATA
# ============================================================

if dataframes:

    st.header("📊 Uploaded Data")

    cols = st.columns(
        len(dataframes)
    )

    for i, (name, df) in enumerate(
        dataframes.items()
    ):

        with cols[i]:

            st.metric(
                name,
                f"{len(df)} rows"
            )

            st.caption(
                f"{len(df.columns)} columns"
            )

    with st.expander(
        "View uploaded tables"
    ):

        for name, df in dataframes.items():

            st.write(
                f"### {name}"
            )

            st.dataframe(
                df,
                use_container_width=True
            )


# ============================================================
# 16. DATA QUALITY
# ============================================================

if dataframes:

    st.header("🔍 Data Quality")

    quality = []

    for name, df in dataframes.items():

        quality.append({

            "Table": name,

            "Rows": len(df),

            "Columns": len(df.columns),

            "Missing": int(
                df.isnull().sum().sum()
            ),

            "Duplicates": int(
                df.duplicated().sum()
            )

        })

    st.dataframe(
        pd.DataFrame(quality),
        use_container_width=True
    )

    warnings = get_warnings(
        dataframes
    )

    if warnings:

        st.warning(
            "Potential data-quality issues detected."
        )

        for warning in warnings:

            st.write(
                "⚠️",
                warning
            )

    else:

        st.success(
            "No obvious data-quality problems detected."
        )


# ============================================================
# 17. ASK QUESTION
# ============================================================

if dataframes:

    st.header("💬 Ask Your Data")

    question = st.text_input(
        "Your question",
        value=st.session_state.question,
        placeholder=(
            "Which employee generated the highest sales?"
        )
    )

    analyze = st.button(
        "🚀 Analyze",
        type="primary"
    )


    # ========================================================
    # 18. ANALYZE
    # ========================================================

    if analyze:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            st.stop()

        # Clear previous result

        st.session_state.question = question
        st.session_state.plan = ""
        st.session_state.code = ""
        st.session_state.result = None
        st.session_state.verified = False
        st.session_state.run_count = 0
        st.session_state.analysis_done = False
        st.session_state.cannot_answer = False

        summary = create_data_summary(
            dataframes
        )

        # ----------------------------------------------------
        # AI PLANNING
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Understanding the question..."
        ):

            try:

                plan = create_plan(
                    question,
                    summary
                )

                st.session_state.plan = plan

            except Exception as e:

                st.error(
                    f"Planning error: {e}"
                )

                st.stop()


        # ----------------------------------------------------
        # CANNOT ANSWER
        # ----------------------------------------------------

        if (
            "CANNOT_ANSWER"
            in plan.upper()
        ):

            st.session_state.cannot_answer = True

            st.error(
                "🔴 CANNOT DETERMINE"
            )

            st.write(
                "The available data does not support "
                "a reliable answer."
            )

            st.subheader(
                "Why?"
            )

            st.code(
                plan,
                language="text"
            )

            st.stop()


        # ----------------------------------------------------
        # SHOW PLAN
        # ----------------------------------------------------

        st.subheader(
            "🧠 Agent Reasoning"
        )

        st.code(
            plan,
            language="text"
        )


        # ----------------------------------------------------
        # GENERATE PROOF
        # ----------------------------------------------------

        with st.spinner(
            "💻 Generating executable proof..."
        ):

            try:

                code = generate_code(
                    question,
                    plan,
                    summary
                )

                st.session_state.code = code

            except Exception as e:

                st.error(
                    f"Code generation error: {e}"
                )

                st.stop()


        # ----------------------------------------------------
        # FIRST EXECUTION
        # ----------------------------------------------------

        with st.spinner(
            "▶ Running proof..."
        ):

            success, result, message = execute_proof(
                code,
                dataframes
            )

        if not success:

            st.error(
                "❌ Proof execution failed."
            )

            st.code(
                message,
                language="text"
            )

            st.subheader(
                "Generated Proof"
            )

            st.code(
                code,
                language="python"
            )

            st.stop()


        st.session_state.result = result
        st.session_state.run_count = 1
        st.session_state.analysis_done = True


# ============================================================
# 19. SHOW ANSWER + PROOF
# ============================================================

if (
    st.session_state.analysis_done
    and not st.session_state.cannot_answer
):

    st.divider()

    st.header("🎯 Answer")

    show_result(
        st.session_state.result
    )


    # ========================================================
    # PROOF
    # ========================================================

    st.header(
        "🧾 Executable Proof"
    )

    st.write(
        "The answer above was produced by executing "
        "the following Python code on your uploaded data."
    )

    st.code(
        st.session_state.code,
        language="python"
    )


    # ========================================================
    # IMPORTANT VERIFIER SECTION
    # ========================================================

    st.header(
        "🔎 Verify This Answer"
    )

    st.info(
        "The verifier can re-run the exact proof code "
        "against the uploaded data."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Proof executions",
            st.session_state.run_count
        )

    with col2:

        if st.session_state.verified:

            st.success(
                "VERIFIED"
            )

        else:

            st.warning(
                "Not independently re-run yet"
            )


    # ========================================================
    # RE-RUN BUTTON
    # ========================================================

    rerun = st.button(
        "▶ Re-run Proof",
        type="primary"
    )


    if rerun:

        with st.spinner(
            "🔁 Re-running the exact proof..."
        ):

            success, new_result, message = execute_proof(
                st.session_state.code,
                dataframes
            )

        if not success:

            st.error(
                "❌ Re-run failed."
            )

            st.code(
                message,
                language="text"
            )

        else:

            st.session_state.run_count += 1

            old_result = (
                st.session_state.result
            )

            matches = compare_results(
                old_result,
                new_result
            )

            st.subheader(
                "Re-run Result"
            )

            show_result(
                new_result
            )

            if matches:

                st.session_state.verified = True

                st.success(
                    "🟢 VERIFIED — Re-run produced the same result."
                )

            else:

                st.session_state.verified = False

                st.error(
                    "🔴 VERIFICATION FAILED — "
                    "The re-run produced a different result."
                )

            # ------------------------------------------------
            # Verification details
            # ------------------------------------------------

            st.subheader(
                "Verification Details"
            )

            verification = pd.DataFrame({

                "Check": [
                    "Original result",
                    "Re-run result",
                    "Results match",
                    "Proof executions"
                ],

                "Value": [
                    str(old_result),
                    str(new_result),
                    str(matches),
                    str(st.session_state.run_count)
                ]

            })

            st.dataframe(
                verification,
                use_container_width=True
            )


    # ========================================================
    # EVIDENCE
    # ========================================================

    st.header(
        "📌 Evidence"
    )

    st.write(
        "Source data used by the proof:"
    )

    for name, df in dataframes.items():

        st.write(
            f"• {name}: "
            f"{len(df)} rows × "
            f"{len(df.columns)} columns"
        )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "🔧 Technical Details"
    ):

        st.write(
            "**Question:**"
        )

        st.write(
            st.session_state.question
        )

        st.write(
            "**Model:**"
        )

        st.write(
            MODEL
        )

        st.write(
            "**Verification method:**"
        )

        st.write(
            "The exact generated proof is executed again "
            "against fresh copies of the uploaded DataFrames."
        )

        st.write(
            "**Execution count:**"
        )

        st.write(
            st.session_state.run_count
        )


# ============================================================
# 20. EMPTY STATE
# ============================================================

else:

    if not dataframes:

        st.info(
            "👈 Upload CSV files to begin."
        )
        