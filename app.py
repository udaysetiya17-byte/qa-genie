from flask import (
    Flask, render_template, request, send_file,
    session, redirect, url_for
)
from dotenv import load_dotenv
from google import genai
import os
import json
import io
import time
import uuid
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "qa-genie-local-secret"
)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing in your .env file."
    )

client = genai.Client(api_key=api_key)

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "qa_data.json"
)

ALLOWED_CASE_COUNTS = [25, 50, 75, 100]

MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash"
]


# ============================================================
# SESSION-WISE STORAGE
# ============================================================

def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    return session["session_id"]


def load_all_data():
    if not os.path.exists(DATA_FILE):
        return {"sessions": {}}

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if (
            isinstance(data, dict)
            and isinstance(data.get("sessions"), dict)
        ):
            return data

        return {"sessions": {}}

    except (json.JSONDecodeError, OSError):
        return {"sessions": {}}


def get_session_data():
    data = load_all_data()
    session_id = get_session_id()

    if session_id not in data["sessions"]:
        data["sessions"][session_id] = {
            "test_cases": [],
            "execution_results": []
        }

    return data, data["sessions"][session_id]


def save_session_data(test_cases, execution_results):
    data = load_all_data()
    session_id = get_session_id()

    data["sessions"][session_id] = {
        "test_cases": test_cases,
        "execution_results": execution_results
    }

    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# TEST CASE GENERATION
# ============================================================

def generate_test_cases(
    requirement,
    test_type="All Testing",
    priority="All Priorities",
    number_of_cases=25
):
    if number_of_cases not in ALLOWED_CASE_COUNTS:
        number_of_cases = 25

    prompt = f"""
You are a Senior QA Engineer.

Analyze the client requirement below and generate practical,
clear, executable software QA test cases.

CLIENT REQUIREMENT:
{requirement}

TEST TYPE:
{test_type}

PRIORITY FILTER:
{priority}

REQUESTED NUMBER OF TEST CASES:
{number_of_cases}

Cover relevant scenarios such as:
1. Positive scenarios
2. Negative scenarios
3. Boundary and edge cases
4. Field validation
5. Error handling
6. Permissions and authorization
7. Security
8. Functional and regression testing
9. API, integration, web or mobile testing when relevant

Do not create duplicate or unrelated test cases.
Use the selected test type and priority filter.

Return ONLY a valid JSON array. Do not include Markdown fences
or explanations before or after the JSON.

Each item must contain these fields:
- Test Case ID
- Test Scenario
- Preconditions
- Test Steps
- Expected Result
- Priority
- Test Type

Use IDs TC001, TC002, TC003, and so on.

Generate exactly {number_of_cases} test cases where the
requirement provides enough scope. Do not invent unrelated
functionality merely to reach the requested count.

Example item:
{{
  "Test Case ID": "TC001",
  "Test Scenario": "Verify the expected behaviour",
  "Preconditions": "User has access to the application",
  "Test Steps": "1. Open the application. 2. Perform the action.",
  "Expected Result": "The application behaves as expected.",
  "Priority": "High",
  "Test Type": "Functional"
}}
"""

    last_error = None

    for model in MODELS:
        for attempt in range(2):
            try:
                print(
                    f"Trying Gemini model: {model}; "
                    f"attempt {attempt + 1}/2"
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                result = (response.text or "").strip()

                if not result:
                    raise ValueError(
                        "Gemini returned an empty response."
                    )

                # Remove optional Markdown code fences.
                if result.startswith("```"):
                    result = result.replace("```json", "")
                    result = result.replace("```JSON", "")
                    result = result.replace("```", "").strip()

                # Extract the JSON array if extra text is returned.
                start = result.find("[")
                end = result.rfind("]")

                if start == -1 or end == -1 or end < start:
                    raise ValueError(
                        "Gemini did not return a valid JSON array."
                    )

                result = result[start:end + 1]
                test_cases = json.loads(result)

                if not isinstance(test_cases, list):
                    raise ValueError(
                        "Gemini response is not a JSON array."
                    )

                cleaned_cases = []

                for index, test_case in enumerate(test_cases):
                    if not isinstance(test_case, dict):
                        continue

                    cleaned_cases.append({
                        "Test Case ID": test_case.get(
                            "Test Case ID",
                            f"TC{index + 1:03d}"
                        ),
                        "Test Scenario": test_case.get(
                            "Test Scenario", ""
                        ),
                        "Preconditions": test_case.get(
                            "Preconditions", ""
                        ),
                        "Test Steps": test_case.get(
                            "Test Steps", ""
                        ),
                        "Expected Result": test_case.get(
                            "Expected Result", ""
                        ),
                        "Priority": test_case.get(
                            "Priority", "Medium"
                        ),
                        "Test Type": test_case.get(
                            "Test Type", test_type
                        )
                    })

                if not cleaned_cases:
                    raise ValueError(
                        "No valid test cases were returned."
                    )

                # Prevent more cases than the user requested.
                cleaned_cases = cleaned_cases[:number_of_cases]

                print(
                    f"Generated {len(cleaned_cases)} test cases "
                    f"using {model}"
                )

                return cleaned_cases

            except Exception as error:
                last_error = error
                print(f"Model {model} failed: {error}")

                if attempt == 0:
                    time.sleep(2)

    raise ValueError(
        "Unable to generate test cases right now. "
        "Please try again in a few seconds.\n\n"
        f"Technical details: {last_error}"
    )


# ============================================================
# HOME / GENERATE PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():
    _, current_data = get_session_data()

    test_cases = current_data.get("test_cases", [])

    requirement = session.get("requirement", "")
    test_type = session.get("test_type", "All Testing")
    priority = session.get("priority", "All Priorities")
    number_of_cases = session.get("number_of_cases", 25)
    error = ""

    if request.method == "POST":
        requirement = request.form.get(
            "requirement", ""
        ).strip()

        test_type = request.form.get(
            "test_type", "All Testing"
        )

        priority = request.form.get(
            "priority", "All Priorities"
        )

        # Accept the new field name and the old one for compatibility.
        submitted_count = request.form.get(
            "number_of_cases",
            request.form.get("case_count", "25")
        )

        try:
            number_of_cases = int(submitted_count)
        except (TypeError, ValueError):
            number_of_cases = 25

        if number_of_cases not in ALLOWED_CASE_COUNTS:
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

                # Save data for this browser session only.
                save_session_data(
                    test_cases=test_cases,
                    execution_results=[]
                )

                # Store only small values in the Flask session.
                session["requirement"] = requirement
                session["test_type"] = test_type
                session["priority"] = priority
                session["number_of_cases"] = number_of_cases

                return redirect(url_for("test_cases_page"))

            except Exception as exception:
                error = str(exception)

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
    _, current_data = get_session_data()

    return render_template(
        "test_cases.html",
        test_cases=current_data.get("test_cases", [])
    )


# ============================================================
# BUILD REPORT DATA
# ============================================================

def build_report_data(execution_results):
    total = len(execution_results)

    passed = sum(
        1 for item in execution_results
        if item.get("Status") == "Pass"
    )

    failed = sum(
        1 for item in execution_results
        if item.get("Status") == "Fail"
    )

    blocked = sum(
        1 for item in execution_results
        if item.get("Status") == "Blocked"
    )

    not_executed = sum(
        1 for item in execution_results
        if item.get("Status") == "Not Executed"
    )

    executed = passed + failed + blocked

    pass_rate = round((passed / total) * 100, 1) if total else 0
    fail_rate = round((failed / total) * 100, 1) if total else 0

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "blocked": blocked,
        "not_executed": not_executed,
        "executed": executed,
        "pass_rate": pass_rate,
        "fail_rate": fail_rate,
        "pass_percentage": pass_rate,
        "fail_percentage": fail_rate,
        "execution_results": execution_results
    }


def make_not_executed_results(test_cases):
    results = []

    for test_case in test_cases:
        results.append({
            "Test Case ID": test_case.get("Test Case ID", ""),
            "Test Scenario": test_case.get("Test Scenario", ""),
            "Priority": test_case.get("Priority", ""),
            "Test Type": test_case.get("Test Type", ""),
            "Status": "Not Executed",
            "Actual Result": "",
            "QA Comments": ""
        })

    return results


# ============================================================
# EXECUTE TEST CASES
# ============================================================

@app.route("/execute", methods=["POST"])
def execute():
    _, current_data = get_session_data()

    test_cases = current_data.get("test_cases", [])

    if not test_cases:
        return (
            "No test cases available. Please generate test cases first.",
            400
        )

    execution_results = []

    for index, test_case in enumerate(test_cases):
        status = request.form.get(
            f"status_{index}",
            "Not Executed"
        )

        actual_result = request.form.get(
            f"actual_result_{index}", ""
        ).strip()

        comments = request.form.get(
            f"comments_{index}", ""
        ).strip()

        execution_results.append({
            "Test Case ID": test_case.get(
                "Test Case ID", f"TC{index + 1:03d}"
            ),
            "Test Scenario": test_case.get("Test Scenario", ""),
            "Priority": test_case.get("Priority", ""),
            "Test Type": test_case.get("Test Type", ""),
            "Status": status,
            "Actual Result": actual_result,
            "QA Comments": comments
        })

    save_session_data(
        test_cases=test_cases,
        execution_results=execution_results
    )

    return render_template(
        "reports.html",
        **build_report_data(execution_results)
    )


# ============================================================
# REPORT PAGE
# ============================================================

@app.route("/reports", methods=["GET"])
def reports():
    _, current_data = get_session_data()

    execution_results = current_data.get(
        "execution_results", []
    )

    if not execution_results:
        execution_results = make_not_executed_results(
            current_data.get("test_cases", [])
        )

    return render_template(
        "reports.html",
        **build_report_data(execution_results)
    )


# ============================================================
# DOWNLOAD TEST CASES AS EXCEL
# ============================================================

@app.route("/download", methods=["POST"])
def download():
    _, current_data = get_session_data()

    submitted_data = request.form.get("test_cases", "")

    if submitted_data:
        try:
            test_cases = json.loads(submitted_data)
        except json.JSONDecodeError:
            return "Invalid test case data.", 400
    else:
        test_cases = current_data.get("test_cases", [])

    if not isinstance(test_cases, list):
        return "Invalid test case format.", 400

    output = io.BytesIO()
    dataframe = pd.DataFrame(test_cases)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(
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
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


# ============================================================
# DOWNLOAD EXECUTION REPORT AS EXCEL
# ============================================================

@app.route("/download-report", methods=["GET"])
def download_report():
    _, current_data = get_session_data()

    execution_results = current_data.get(
        "execution_results", []
    )

    if not execution_results:
        execution_results = make_not_executed_results(
            current_data.get("test_cases", [])
        )

    report = build_report_data(execution_results)

    summary_dataframe = pd.DataFrame({
        "Metric": [
            "Total Test Cases",
            "Passed",
            "Failed",
            "Blocked",
            "Not Executed",
            "Executed",
            "Pass Rate (%)",
            "Fail Rate (%)"
        ],
        "Value": [
            report["total"],
            report["passed"],
            report["failed"],
            report["blocked"],
            report["not_executed"],
            report["executed"],
            report["pass_rate"],
            report["fail_rate"]
        ]
    })

    details_dataframe = pd.DataFrame(execution_results)

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Summary"
        )

        details_dataframe.to_excel(
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
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000))
    )