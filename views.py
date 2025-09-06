from django.shortcuts import render,redirect
import joblib
import google.generativeai as genai
import json
import requests
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

# Load the saved model.pkl once
ml = joblib.load("mentalmate/model.pkl")
model = ml["model"]
vectorizer = ml["vectorizer"]
label_encoder = ml["label_encoder"]
# Suggestions for each predicted condition
SUGGESTIONS = {
    "depression": "1.Try to maintain a routine, 2.do light physical activity, and 3.talk to a trusted person.4. Consider seeing a therapist. 5.Consider talking to a counselor or writing down your thoughts daily.",
    "anxiety": "1.Practice deep breathing, 2.limit caffeine, and take short breaks. 4.Meditation and mindfulness can also help.",
    "stress": "Take breaks, manage your time wisely, and do relaxing activities like walking or listening to music.",
    "PTSD": "Avoid triggers when possible, seek professional therapy, and stay connected with supportive people.",
    "ADHD": "Use task lists, set reminders, and break work into small chunks. A structured routine may help.",
    "none": "Keep maintaining your good mental health! Stay mindful and continue healthy habits.",
    "neutral": "All good! You seem fine. 😊",
}


 # change "home" to your path name if different

'''
def predict_condition(request):
    prediction = None
    if request.method == "POST":
        user_input = request.POST.get("user_text")  # get input from form
        if user_input:
            vec = vectorizer.transform([user_input])
            pred = model.predict(vec)
            prediction = label_encoder.inverse_transform(pred)[0]

    return render(request, "index.html", {"prediction": prediction})
'''


def predict_condition(request):
    if request.method == "POST" and request.POST.get("clear_chat"):
        request.session["chat_history"] = []
        return redirect("predict_condition")

    chat_history = request.session.get("chat_history", [])
    prediction = None
    suggestion = None
    user_input = None
    video_file = None  # Important

    if request.method == "POST" and request.POST.get("user_text"):
        user_input = request.POST["user_text"]
        if len(user_input.strip().split()) < 2:
            prediction = "Neutral"
            suggestion = "Hi there! Feel free to share how you're feeling today. 😊"
        else:
            vec = vectorizer.transform([user_input])
            pred = model.predict(vec)
            prediction = label_encoder.inverse_transform(pred)[0]
            suggestion = SUGGESTIONS.get(prediction, "No suggestion available.")

            # 🧠 YouTube videos mapped to conditions
            video_map = {
                "depression": "https://www.youtube.com/embed/3QIfkeA6HBY",
                "anxiety": "https://www.youtube.com/embed/tPXR-f10MFs",
                "stress": "https://www.youtube.com/embed/WuyPuH9ojCE",
                "PTSD": "https://www.youtube.com/watch?v=2KXtlIX_yUs",
                "ADHD": "https://www.youtube.com/watch?v=1t9UHQgtDfU&t=64s"
            }
            video_file = video_map.get(prediction)

        chat_history.append({
            "user": f"🧍 You: {user_input}",
            "bot": f"🤖 MentalMate: You might be experiencing <b>{prediction}</b>.<br>💡 Suggestion: {suggestion}"
        })
        request.session["chat_history"] = chat_history

    return render(request, "index.html", {
        "chat_history": chat_history,
        "video_file": video_file,
        "prediction": prediction,
        "suggestion": suggestion
    })


API_KEY = 'AIzaSyDv_eIt-WOoA73Gqpvkdw1HcCTtHg1iW_k'
API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}'

@csrf_exempt
def gemini_assistant(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_input = data.get("message", "")

            if not user_input.strip():
                return JsonResponse({"reply": "Please enter a valid question."})

            payload = {
                "contents": [
                    {
                        "parts": [{"text": user_input}]
                    }
                ]
            }

            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
                return JsonResponse({"reply": generated_text})
            else:
                return JsonResponse({
                    "reply": f"Error {response.status_code}: {response.text}"
                })

        except Exception as e:
            return JsonResponse({"reply": f"Exception: {str(e)}"})

    return JsonResponse({"reply": "Only POST method allowed."})






