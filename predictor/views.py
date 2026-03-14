from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib import messages
from django.http import JsonResponse
import joblib
import os
import pandas as pd
from .models import Prediction

# ------------------------------------------------------------------
# ✅ Load model & scaler ONCE at startup — stays in memory forever.
#    This is the fix for the ~1 minute prediction delay.
#    Previously these were inside _run_prediction() and reloaded
#    from disk on every single form submission.
# ------------------------------------------------------------------
_BASE_DIR = os.path.join('predictor', 'ml_model')
model  = joblib.load(os.path.join(_BASE_DIR, 'best_lgb_model.pkl'))
scaler = joblib.load(os.path.join(_BASE_DIR, 'scaler.pkl'))


def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful! You are now logged in.")
            return redirect('dashboard')
    else:
        form = UserCreationForm()

    return render(request, 'predictor/register.html', {'form': form})


# ------------------------------------------------------------------
# Shared helper — save_to_db controls whether result is stored
# ------------------------------------------------------------------
def _run_prediction(request, save_to_db):
    if request.method == 'POST':
        try:
            gender      = request.POST.get('gender', '').lower()
            pregnancies = float(request.POST.get('pregnancies', 0)) if gender != 'male' else 0.0
            glucose                    = float(request.POST.get('glucose', 0))
            blood_pressure             = float(request.POST.get('blood_pressure', 0))
            skin_thickness             = float(request.POST.get('skin_thickness', 0))
            insulin                    = float(request.POST.get('insulin', 0))
            bmi                        = float(request.POST.get('bmi', 0))
            diabetes_pedigree_function = float(request.POST.get('diabetes_pedigree_function', 0))
            age                        = float(request.POST.get('age', 0))
        except ValueError:
            messages.error(request, "Invalid input. Please enter valid numbers.")
            return render(request, "predictor/predict.html")

        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        input_values = [
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree_function, age
        ]
        input_data = pd.DataFrame([input_values], columns=feature_columns)

        # ✅ model and scaler are already in memory — no file I/O here
        input_data_scaled = scaler.transform(input_data)
        y_pred_proba      = model.predict_proba(input_data_scaled)[:, 1]

        optimal_threshold = 0.16
        prediction        = (y_pred_proba >= optimal_threshold).astype(int)
        result            = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        if save_to_db and request.user.is_authenticated:
            Prediction.objects.create(
                user=request.user,
                prediction_result=result
            )

        return render(request, "predictor/result.html", {"result": result})

    # GET — render the blank prediction form
    return render(request, "predictor/predict.html")


@login_required
def predict_diabetes(request):
    return _run_prediction(request, save_to_db=True)


def predict_diabetes_guest(request):
    return _run_prediction(request, save_to_db=False)


@login_required
def user_dashboard(request):
    user_predictions = Prediction.objects.filter(
        user=request.user
    ).order_by('-timestamp')

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        predictions = [
            {
                "timestamp":         p.timestamp.isoformat(),
                "prediction_result": p.prediction_result
            }
            for p in user_predictions
        ]
        return JsonResponse({"predictions": predictions})

    return render(request, "predictor/dashboard.html", {"predictions": user_predictions})