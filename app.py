from flask import Flask, render_template, request, send_file, session, redirect
from dotenv import load_dotenv
from google import genai
import os
import json
import io
import time
import pandas as pd

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

app = Flask(__name__)

# Used to keep generated test cases and execution results
# while the local Flask application is running.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "qa-genie-local-secret")

# ============================================================
# GEMINI CLIENT
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing in .env file")

client = genai.Client(api_key=api_key)

# ============================================================
# LOCAL APP DATA STORE
# ============================================================
# Keep generated test cases and execution results outside the
# Flask session cookie. This avoids oversized browser cookies.
test_cases_store = []
execution_results_store = []

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_data.json")


def load_data():
    global test_cases_store, execution_results_store

    if not os.path.exists(DATA_FILE):
        test_cases_store = []
        execution_results_store = []
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases_store = data.get("test_cases", [])
        execution_results_store = data.get("execution_results", [])

    except (json.JSONDecodeError, OSError):
        test_cases_store = []
        execution_results_store = []


def save_data():
    data = {
        "test_cases": test_cases_store,
        "execution_results": execution_results_store
    }

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


load_data()


# Primary + fallback models
MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash"
]


# ============================================================
# GENERATE TEST CASES
# ============================================================

def generate_test_cases(
    requirement,
    test_type="All Testing",
    priority="All Priorities",
    number_of_cases=25
):
    prompt = f"""
You are a Senior QA Engineer with extensive experience in:

- Web Testing
- Mobile Testing
- API Testing
- Integration Testing
- Security Testing
- Regression Testing
- Functional Testing
- Non-functional Testing

Analyze the following client requirement.

CLIENT REQUIREMENT:
{requirement}

TEST TYPE:
{test_type}

PRIORITY:
{priority}

NUMBER OF TEST CASES:
{number_of_cases}

Generate practical and executable QA test cases.

Cover applicable scenarios including:

1. Positive scenarios
2. Negative scenarios
3. Boundary and edge cases
4. Field validations
5. Error handling
6. Permission and authorization
7. Security
8. Web testing
9. Mobile testing
10. API testing
11. Integration testing
12. Regression testing

Avoid duplicate test cases.

Only generate scenarios that are relevant to the requirement.

Return ONLY a valid JSON array.

Do NOT use markdown.
Do NOT use ```json.
Do NOT add explanations before or after the JSON.

Each test case must have exactly these fields:

[
    {{
        "Test Case ID": "TC001",
        "Test Scenario": "Verify ...",
        "Preconditions": "...",
        "Test Steps": "1. ... 2. ... 3. ...",
        "Expected Result": "...",
        "Priority": "High",
        "Test Type": "Functional"
    }}
]

Generate exactly the requested number of test cases where
the requirement provides enough scope.

If the requirement is very small, generate the most relevant
test cases without inventing unrelated functionality.
"""

    last_error = None

    for model in MODELS:

        for attempt in range(2):

            try:
                print(
                    f"Trying model: {model} "
                    f"(attempt {attempt + 1}/2)"
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                result = response.text.strip()

                if not result:
                    raise ValueError(
                        "Gemini returned an empty response."
                    )

                if result.startswith("```"):
                    result = result.replace("```json", "")
                    result = result.replace("```", "")
                    result = result.strip()

                start = result.find("[")
                end = result.rfind("]")

                if start != -1 and end != -1:
                    result = result[start:end + 1]

                test_cases = json.loads(result)

                if not isinstance(test_cases, list):
                    raise ValueError(
                        "Gemini response is not a JSON array."
                    )

                print(
                    f"Successfully generated "
                    f"{len(test_cases)} test cases "
                    f"using {model}"
                )

                return test_cases

            except Exception as e:
                last_error = e
                print(f"Model {model} failed: {str(e)}")

                if attempt == 0:
                    time.sleep(2)

    raise ValueError(
        "Unable to generate test cases right now. "
        "The AI service may be temporarily unavailable. "
        "Please try again in a few seconds.\n\n"
        f"Technical details: {last_error}"
    )


# ============================================================
# HOME / GENERATE PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    global test_cases_store, execution_results_store

    load_data()
    test_cases = test_cases_store
    requirement = session.get("requirement", "")
    error = ""

    test_type = session.get("test_type", "All Testing")
    priority = session.get("priority", "All Priorities")
    number_of_cases = session.get("number_of_cases", 25)

    if request.method == "POST":

        requirement = request.form.get(
            "requirement", ""
        ).strip()

        test_type = request.form.get(
            "test_type",
            "All Testing"
        )

        priority = request.form.get(
            "priority",
            "All Priorities"
        )

        try:
            number_of_cases = int(
                request.form.get(
                    "number_of_cases",
                    25
                )
            )
        except ValueError:
            number_of_cases = 25

        if not requirement:
            error = "Please enter a client requirement."

        else:
            try:
                test_cases = generate_test_cases(
                    requirement=requirement,
                    test_type=test_type,
                    priority=priority,
                    number_of_cases=number_of_cases
                )

                # Save generated cases in the local app store.
                # Do not put the full test case list in the Flask
                # session because the session is stored in a browser cookie.
                test_cases_store = test_cases
                execution_results_store = []
                save_data()

                # Keep only small form values in the session.
                session["requirement"] = requirement
                session["test_type"] = test_type
                session["priority"] = priority
                session["number_of_cases"] = number_of_cases

                return redirect("/test-cases")

            except Exception as e:
                error = str(e)

    return render_template(
        "index.html",
        requirement=requirement,
        test_cases=test_cases,
        error=error,
        test_type=test_type,
        priority=priority,
        number_of_cases=number_of_cases
    )


# ============================================================
# TEST CASES PAGE
# ============================================================

@app.route("/test-cases", methods=["GET"])
def test_cases_page():

    load_data()
    test_cases = test_cases_store

    return render_template(
        "test_cases.html",
        test_cases=test_cases
    )


# ============================================================
# EXECUTE TEST CASES
# ============================================================

@app.route("/execute", methods=["POST"])
def execute():

    global execution_results_store

    load_data()
    test_cases = test_cases_store

    if not test_cases:
        return "No test cases available. Please generate test cases first.", 400

    execution_results = []

    for index, tc in enumerate(test_cases):

        status = request.form.get(
            f"status_{index}",
            "Not Executed"
        )

        actual_result = request.form.get(
            f"actual_result_{index}",
            ""
        ).strip()

        comments = request.form.get(
            f"comments_{index}",
            ""
        ).strip()

        execution_results.append({
            "Test Case ID": tc.get("Test Case ID", f"TC{index + 1:03d}"),
            "Test Scenario": tc.get("Test Scenario", ""),
            "Priority": tc.get("Priority", ""),
            "Test Type": tc.get("Test Type", ""),
            "Status": status,
            "Actual Result": actual_result,
            "QA Comments": comments
        })

    execution_results_store = execution_results
    save_data()

    return render_template(
        "reports.html",
        **build_report_data(execution_results)
    )


# ============================================================
# BUILD REPORT DATA
# ============================================================

def build_report_data(execution_results):

    total = len(execution_results)

    passed = sum(
        1 for item in execution_results
        if item["Status"] == "Pass"
    )

    failed = sum(
        1 for item in execution_results
        if item["Status"] == "Fail"
    )

    blocked = sum(
        1 for item in execution_results
        if item["Status"] == "Blocked"
    )

    not_executed = sum(
        1 for item in execution_results
        if item["Status"] == "Not Executed"
    )

    executed = passed + failed + blocked

    if total > 0:
        pass_rate = round((passed / total) * 100, 1)
        fail_rate = round((failed / total) * 100, 1)
    else:
        pass_rate = 0
        fail_rate = 0

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "blocked": blocked,
        "not_executed": not_executed,
        "executed": executed,
        "pass_rate": pass_rate,
        "fail_rate": fail_rate,

        # These names match the current reports.html you showed.
        "pass_percentage": pass_rate,
        "fail_percentage": fail_rate,

        "execution_results": execution_results
    }


# ============================================================
# REPORT PAGE
# ============================================================

@app.route("/reports", methods=["GET"])
def reports():

    load_data()
    execution_results = execution_results_store

    # If execution has not happened yet, show all generated
    # test cases as Not Executed.
    if not execution_results:

        test_cases = test_cases_store

        execution_results = []

        for tc in test_cases:
            execution_results.append({
                "Test Case ID": tc.get("Test Case ID", ""),
                "Test Scenario": tc.get("Test Scenario", ""),
                "Priority": tc.get("Priority", ""),
                "Test Type": tc.get("Test Type", ""),
                "Status": "Not Executed",
                "Actual Result": "",
                "QA Comments": ""
            })

    return render_template(
        "reports.html",
        **build_report_data(execution_results)
    )


# ============================================================
# DOWNLOAD TEST CASES AS EXCEL
# ============================================================

@app.route("/download", methods=["POST"])
def download():

    load_data()
    data = request.form.get("test_cases")

    if not data:
        # Fallback to the local app store if the form does not send the data.
        test_cases = test_cases_store
    else:
        try:
            test_cases = json.loads(data)
        except json.JSONDecodeError:
            return "Invalid test case data.", 400

    if not isinstance(test_cases, list):
        return "Invalid test case format.", 400

    df = pd.DataFrame(test_cases)

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Test Cases"
        )

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="AI_Test_Cases.xlsx",
        mimetype=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        )
    )


# ============================================================
# DOWNLOAD QA EXECUTION REPORT AS EXCEL
# ============================================================

@app.route("/download-report", methods=["GET"])
def download_report():

    load_data()
    execution_results = execution_results_store

    if not execution_results:
        test_cases = test_cases_store

        execution_results = []

        for tc in test_cases:
            execution_results.append({
                "Test Case ID": tc.get("Test Case ID", ""),
                "Test Scenario": tc.get("Test Scenario", ""),
                "Priority": tc.get("Priority", ""),
                "Test Type": tc.get("Test Type", ""),
                "Status": "Not Executed",
                "Actual Result": "",
                "QA Comments": ""
            })

    report = build_report_data(execution_results)

    summary_df = pd.DataFrame({
        "Metric": [
            "Total Test Cases",
            "Passed",
            "Failed",
            "Blocked",
            "Not Executed",
            "Executed",
            "Pass Rate",
            "Fail Rate"
        ],
        "Count": [
            report["total"],
            report["passed"],
            report["failed"],
            report["blocked"],
            report["not_executed"],
            report["executed"],
            f'{report["pass_rate"]}%',
            f'{report["fail_rate"]}%'
        ]
    })

    details_df = pd.DataFrame(execution_results)

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            index=False,
            sheet_name="QA Summary"
        )

        details_df.to_excel(
            writer,
            index=False,
            sheet_name="Execution Details"
        )

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="QA_Execution_Report.xlsx",
        mimetype=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        )
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
