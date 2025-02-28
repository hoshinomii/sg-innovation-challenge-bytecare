from google import genai
import sys

client  = genai.Client(api_key="ENTER YOUR API KEY HERE")
content = sys.argv[1]
response = client.models.generate_content(
    model ="gemini-2.0-flash", contents=content
)

print(response.text)

