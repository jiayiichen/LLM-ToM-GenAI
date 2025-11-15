curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
-H "Content-Type: application/json" \
https://aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/global/publishers/google/models/${MODEL_ID}:streamGenerateContent -d \
$'{
  "contents": {
    "role": "user",
    "parts": [
      {
        "text": "Which of the following is a planet? A Moon B Jupiter C Sun Answer with the letter of the correct option."
      }
    ]
  }
}'
