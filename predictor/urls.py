from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

# Define all application routes
urlpatterns = [
    # Default login route (home page)
    path('', auth_views.LoginView.as_view(template_name='predictor/login.html'), name='login'),

    # Login route (explicit, can be accessed directly via /login/)
    path('login/', auth_views.LoginView.as_view(template_name='predictor/login.html'), name='login'),

    # Logout route
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # User registration route
    path('register/', views.register_view, name='register'),

    # User dashboard (history of predictions)
    path('dashboard/', views.user_dashboard, name='dashboard'),

    # Diabetes prediction form route
    path('predict/', views.predict_diabetes, name='predict_diabetes'),
    path('predict/guest/', views.predict_diabetes_guest, name='predict_diabetes_guest'),  # ← added

]
