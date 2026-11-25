import joblib

model = joblib.load("ticket_priority_model.joblib")

ticket = [
    "Server is down and all users cannot access the application"
]

prediction = model.predict(ticket)

print("Predicted Priority:", prediction[0])