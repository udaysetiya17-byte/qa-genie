import os
from dotenv import load_dotenv
from google import genai

# Load .env
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env file")

# Create Gemini client
client = genai.Client(api_key=api_key)


def generate_test_cases(requirement):

    prompt = f"""
You are an expert Senior QA Engineer.

Analyze the following client requirement and generate comprehensive
software test cases.

CLIENT REQUIREMENT:
{requirement}

Cover the following areas where applicable:

1. Positive scenarios
2. Negative scenarios
3. Boundary and edge cases
4. Field validations
5. Error handling
6. API testing
7. Permission and security scenarios
8. Web testing
9. Mobile testing
10. Integration testing
11. Regression scenarios

For every test case provide:

- Test Case ID
- Test Scenario
- Preconditions
- Test Steps
- Expected Result
- Priority
- Test Type

Do not create duplicate test cases.

Make the test cases practical and detailed enough for a QA engineer
to execute manually.
"""

    # Create chat
    chat = client.chats.create(
        model="gemini-3.8-flash"
    )

    # Send requirement to Gemini
    response = chat.send_message(
        message=prompt
    )

    return response.text


print("\n===== AI TEST CASE GENERATOR =====\n")

requirement = input("Enter client requirement:\n")

print("\nGenerating test cases...\n")

result = generate_test_cases(requirement)

print("\n===== GENERATED TEST CASES =====\n")
print(result)