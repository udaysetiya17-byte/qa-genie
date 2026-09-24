from flask import Flask, render_template, request, send_file, session, redirect
from dotenv import load_dotenv
from google import genai
import os
import json
import io
import time
import uuid
import pandas as pd


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

app = Flask(__name__)

# Secret key is required for Flask sessions.
# For Render, add FLASK_SECRET_KEY as an Environment Variable.
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "qa-genie-local-secret"
)


# ============================================================
# GEMINI API
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing in .env file"
    )

client = genai.Client(api_key=api_key)


# ============================================================
# SESSION-WISE DATA STORAGE
# ============================================================

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "qa_data.json"
)


def get_session_id():
    """
    Create a unique session ID for each browser.
    """

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    return session["session_id"]


def load_all_data():
    """
    Load all session data from qa_data.json.
    """

    if not os.path.exists(DATA_FILE):
        return {
            "sessions": {}
        }

    try:

        with open(
            DATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        # New session-based structure
        if (
            isinstance(data, dict)
            and "sessions" in data
            and isinstance(data["sessions"], dict)
        ):
            return data

        # Ignore old shared data structure.
        return {
            "sessions": {}
        }

    except (
        json.JSONDecodeError,
        OSError
    ):

        return {
            "sessions": {}
        }


def get_session_data():
    """
    Return the data belonging ONLY to the
    current browser session.
    """

    data = load_all_data()

    session_id = get_session_id()

    if session_id not in data["sessions"]:

        data["sessions"][session_id] = {
            "test_cases": [],
            "execution_results": []
        }

    return (
        data,
        data["sessions"][session_id]
    )


def save_session_data(
    test_cases,
    execution_results
):
    """
    Save test cases and execution results
    only for the current browser session.
    """

    data = load_all_data()

    session_id = get_session_id()

    data["sessions"][session_id] = {

        "test_cases": test_cases,

        "execution_results": execution_results

    }

    with open(
        DATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# GEMINI MODELS
# ============================================================

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

Only generate scenarios that are relevant
to the requirement.

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

Generate exactly the requested number of test cases
where the requirement provides enough scope.

If the requirement is very small, generate the most
relevant test cases without inventing unrelated functionality.

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


                # Remove markdown fences
                if result.startswith("```"):

                    result = result.replace(
                        "```json",
                        ""
                    )

                    result = result.replace(
                        "```",
                        ""
                    )

                    result = result.strip()


                # Extract JSON array
                start = result.find("[")

                end = result.rfind("]")


                if (
                    start != -1
                    and end != -1
                ):

                    result = result[
                        start:end + 1
                    ]


                test_cases = json.loads(
                    result
                )


                if not isinstance(
                    test_cases,
                    list
                ):

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

                print(
                    f"Model {model} failed: {str(e)}"
                )

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

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    _, current_data = get_session_data()


    test_cases = current_data.get(
        "test_cases",
        []
    )


    requirement = session.get(
        "requirement",
        ""
    )


    error = ""


    test_type = session.get(
        "test_type",
        "All Testing"
    )


    priority = session.get(
        "priority",
        "All Priorities"
    )


    number_of_cases = session.get(
        "number_of_cases",
        25
    )


    if request.method == "POST":

        requirement = request.form.get(
            "requirement",
            ""
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

            error = (
                "Please enter a client requirement."
            )


        else:

            try:

                test_cases = generate_test_cases(

                    requirement=requirement,

                    test_type=test_type,

                    priority=priority,

                    number_of_cases=number_of_cases

                )


                # Save ONLY for this browser
                save_session_data(

                    test_cases=test_cases,

                    execution_results=[]

                )


                session["requirement"] = (
                    requirement
                )

                session["test_type"] = (
                    test_type
                )

                session["priority"] = (
                    priority
                )

                session["number_of_cases"] = (
                    number_of_cases
                )


                return redirect(
                    "/test-cases"
                )


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

@app.route(
    "/test-cases",
    methods=["GET"]
)
def test_cases_page():

    _, current_data = (
        get_session_data()
    )


    test_cases = current_data.get(
        "test_cases",
        []
    )


    return render_template(

        "test_cases.html",

        test_cases=test_cases

    )


# ============================================================
# EXECUTE TEST CASES
# ============================================================

@app.route(
    "/execute",
    methods=["POST"]
)
def execute():

    _, current_data = (
        get_session_data()
    )


    test_cases = current_data.get(
        "test_cases",
        []
    )


    if not test_cases:

        return (
            "No test cases available. "
            "Please generate test cases first.",
            400
        )


    execution_results = []


    for index, tc in enumerate(
        test_cases
    ):

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

            "Test Case ID": tc.get(

                "Test Case ID",

                f"TC{index + 1:03d}"

            ),

            "Test Scenario": tc.get(

                "Test Scenario",

                ""

            ),

            "Priority": tc.get(

                "Priority",

                ""

            ),

            "Test Type": tc.get(

                "Test Type",

                ""

            ),

            "Status": status,

            "Actual Result": actual_result,

            "QA Comments": comments

        })


    # Save only for current session
    save_session_data(

        test_cases=test_cases,

        execution_results=execution_results

    )


    return render_template(

        "reports.html",

        **build_report_data(
            execution_results
        )

    )


# ============================================================
# REPORT CALCULATION
# ============================================================

def build_report_data(
    execution_results
):

    total = len(
        execution_results
    )


    passed = sum(

        1

        for item in execution_results

        if item.get("Status")
        == "Pass"

    )


    failed = sum(

        1

        for item in execution_results

        if item.get("Status")
        == "Fail"

    )


    blocked = sum(

        1

        for item in execution_results

        if item.get("Status")
        == "Blocked"

    )


    not_executed = sum(

        1

        for item in execution_results

        if item.get("Status")
        == "Not Executed"

    )


    executed = (

        passed
        + failed
        + blocked

    )


    if total > 0:

        pass_rate = round(

            (passed / total) * 100,

            1

        )


        fail_rate = round(

            (failed / total) * 100,

            1

        )

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

        "pass_percentage": pass_rate,

        "fail_percentage": fail_rate,

        "execution_results":
            execution_results

    }


# ============================================================
# REPORT PAGE
# ============================================================

@app.route(
    "/reports",
    methods=["GET"]
)
def reports():

    _, current_data = (
        get_session_data()
    )


    execution_results = (
        current_data.get(
            "execution_results",
            []
        )
    )


    # If execution has not happened yet,
    # show all test cases as Not Executed.

    if not execution_results:

        test_cases = current_data.get(

            "test_cases",

            []

        )


        execution_results = []


        for tc in test_cases:

            execution_results.append({

                "Test Case ID": tc.get(

                    "Test Case ID",

                    ""

                ),

                "Test Scenario": tc.get(

                    "Test Scenario",

                    ""

                ),

                "Priority": tc.get(

                    "Priority",

                    ""

                ),

                "Test Type": tc.get(

                    "Test Type",

                    ""

                ),

                "Status": "Not Executed",

                "Actual Result": "",

                "QA Comments": ""

            })


    return render_template(

        "reports.html",

        **build_report_data(
            execution_results
        )

    )


# ============================================================
# DOWNLOAD TEST CASES EXCEL
# ============================================================

@app.route(
    "/download",
    methods=["POST"]
)
def download():

    _, current_data = (
        get_session_data()
    )


    data = request.form.get(
        "test_cases"
    )


    if not data:

        test_cases = current_data.get(

            "test_cases",

            []

        )

    else:

        try:

            test_cases = json.loads(
                data
            )

        except json.JSONDecodeError:

            return (
                "Invalid test case data.",
                400
            )


    if not isinstance(
        test_cases,
        list
    ):

        return (
            "Invalid test case format.",
            400
        )


    df = pd.DataFrame(
        test_cases
    )


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

        download_name=(
            "AI_Test_Cases.xlsx"
        ),

        mimetype=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        )

    )


# ============================================================
# DOWNLOAD QA EXECUTION REPORT
# ============================================================

@app.route(
    "/download-report",
    methods=["GET"]
)
def download_report():

    _, current_data = (
        get_session_data()
    )


    execution_results = (
        current_data.get(
            "execution_results",
            []
        )
    )


    if not execution_results:

        test_cases = current_data.get(

            "test_cases",

            []

        )


        execution_results = []


        for tc in test_cases:

            execution_results.append({

                "Test Case ID": tc.get(

                    "Test Case ID",

                    ""

                ),

                "Test Scenario": tc.get(

                    "Test Scenario",

                    ""

                ),

                "Priority": tc.get(

                    "Priority",

                    ""

                ),

                "Test Type": tc.get(

                    "Test Type",

                    ""

                ),

                "Status": "Not Executed",

                "Actual Result": "",

                "QA Comments": ""

            })


    report = build_report_data(
        execution_results
    )


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


    details_df = pd.DataFrame(
        execution_results
    )


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

        download_name=(
            "QA_Execution_Report.xlsx"
        ),

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