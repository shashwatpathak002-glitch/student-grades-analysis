import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import hashlib
import json
import os
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Student Grades Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for authentication
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'user_data' not in st.session_state:
    st.session_state['user_data'] = pd.DataFrame()

# Simple user database (in production, use proper database)
USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_page():
    st.markdown('<h1 style="text-align: center; color: #1E88E5;">📊 Student Grades Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center;">Login / Register</h3>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔑 Login", "✨ Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("🔓 Login", use_container_width=True):
            users = load_users()
            if login_username in users:
                if users[login_username]['password'] == hash_password(login_password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = login_username
                    st.success(f"Welcome back, {login_username}!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
            else:
                st.error("❌ Username not found")
    
    with tab2:
        st.subheader("Create New Account")
        reg_username = st.text_input("Choose Username", key="reg_user")
        reg_password = st.text_input("Choose Password", type="password", key="reg_pass")
        reg_password2 = st.text_input("Confirm Password", type="password", key="reg_pass2")
        
        if st.button("✨ Register", use_container_width=True):
            if not reg_username or not reg_password:
                st.error("⚠️ Please fill all fields")
            elif reg_password != reg_password2:
                st.error("⚠️ Passwords don't match")
            else:
                users = load_users()
                if reg_username in users:
                    st.error("⚠️ Username already exists")
                else:
                    users[reg_username] = {
                        'password': hash_password(reg_password),
                        'created_at': str(datetime.now())
                    }
                    save_users(users)
                    st.success("✅ Account created! Please login.")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Check if user is logged in
if not st.session_state['logged_in']:
    login_page()
    st.stop()

# Title
st.markdown('<h1 class="main-header">📊 Student Grades Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align: center; color: gray;">Created by Shashwat Pathak | Welcome, {st.session_state["username"]}</p>', unsafe_allow_html=True)

# Logout button in sidebar
if st.sidebar.button("🚪 Logout"):
    st.session_state['logged_in'] = False
    st.session_state['username'] = ''
    st.session_state['user_data'] = pd.DataFrame()
    st.rerun()

st.markdown("---")

# Function to generate sample dataset
def generate_sample_data(n_students=500):
    np.random.seed(42)
    
    student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, n_students + 1)]
    names = [f"Student_{i}" for i in range(1, n_students + 1)]
    genders = np.random.choice(['Male', 'Female'], n_students)
    
    # Factors affecting performance
    study_hours = np.random.normal(5, 2, n_students).clip(0, 12)
    attendance = np.random.normal(75, 15, n_students).clip(40, 100)
    parent_education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_students, p=[0.3, 0.4, 0.2, 0.1])
    internet_access = np.random.choice(['Yes', 'No'], n_students, p=[0.7, 0.3])
    extra_classes = np.random.choice(['Yes', 'No'], n_students, p=[0.4, 0.6])
    
    # Subject marks (influenced by factors)
    base_performance = study_hours * 5 + attendance * 0.3
    
    subjects = {
        'Mathematics': base_performance + np.random.normal(0, 10, n_students),
        'Science': base_performance + np.random.normal(2, 10, n_students),
        'English': base_performance + np.random.normal(-2, 12, n_students),
        'History': base_performance + np.random.normal(0, 15, n_students),
        'Computer_Science': base_performance + np.random.normal(5, 8, n_students)
    }
    
    df = pd.DataFrame({
        'Student_ID': student_ids,
        'Name': names,
        'Gender': genders,
        'Study_Hours_Per_Day': np.round(study_hours, 1),
        'Attendance_Percentage': np.round(attendance, 1),
        'Parent_Education': parent_education,
        'Internet_Access': internet_access,
        'Extra_Classes': extra_classes
    })
    
    for subject, marks in subjects.items():
        df[subject] = np.round(marks.clip(0, 100), 1)
    
    df['Total_Marks'] = df[['Mathematics', 'Science', 'English', 'History', 'Computer_Science']].sum(axis=1)
    df['Average_Marks'] = np.round(df['Total_Marks'] / 5, 2)
    df['Grade'] = pd.cut(df['Average_Marks'], bins=[0, 40, 50, 60, 70, 80, 100], labels=['F', 'D', 'C', 'B', 'A', 'A+'])
    df['Pass_Status'] = df['Average_Marks'].apply(lambda x: 'Pass' if x >= 40 else 'Fail')
    
    return df

# Manual Data Entry Form
def add_student_manually():
    st.subheader("➕ Add Student Data Manually")
    
    with st.form("student_entry_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            student_id = st.text_input("Student ID", value=f"STU{len(st.session_state['user_data'])+1:04d}")
            name = st.text_input("Student Name")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        with col2:
            study_hours = st.number_input("Study Hours Per Day", 0.0, 12.0, 5.0, 0.5)
            attendance = st.number_input("Attendance %", 0.0, 100.0, 75.0, 1.0)
            parent_edu = st.selectbox("Parent Education", ["High School", "Bachelor", "Master", "PhD"])
        
        with col3:
            internet = st.selectbox("Internet Access", ["Yes", "No"])
            extra_class = st.selectbox("Extra Classes", ["Yes", "No"])
        
        st.markdown("### Subject Marks")
        col4, col5, col6, col7, col8 = st.columns(5)
        
        with col4:
            math = st.number_input("Mathematics", 0.0, 100.0, 70.0, 0.5)
        with col5:
            science = st.number_input("Science", 0.0, 100.0, 70.0, 0.5)
        with col6:
            english = st.number_input("English", 0.0, 100.0, 70.0, 0.5)
        with col7:
            history = st.number_input("History", 0.0, 100.0, 70.0, 0.5)
        with col8:
            cs = st.number_input("Computer Science", 0.0, 100.0, 70.0, 0.5)
        
        submitted = st.form_submit_button("✅ Add Student", use_container_width=True)
        
        if submitted:
            total = math + science + english + history + cs
            average = total / 5
            
            if average >= 80:
                grade = 'A+'
            elif average >= 70:
                grade = 'A'
            elif average >= 60:
                grade = 'B'
            elif average >= 50:
                grade = 'C'
            elif average >= 40:
                grade = 'D'
            else:
                grade = 'F'
            
            pass_status = 'Pass' if average >= 40 else 'Fail'
            
            new_student = {
                'Student_ID': student_id,
                'Name': name,
                'Gender': gender,
                'Study_Hours_Per_Day': study_hours,
                'Attendance_Percentage': attendance,
                'Parent_Education': parent_edu,
                'Internet_Access': internet,
                'Extra_Classes': extra_class,
                'Mathematics': math,
                'Science': science,
                'English': english,
                'History': history,
                'Computer_Science': cs,
                'Total_Marks': total,
                'Average_Marks': round(average, 2),
                'Grade': grade,
                'Pass_Status': pass_status
            }
            
            st.session_state['user_data'] = pd.concat([
                st.session_state['user_data'],
                pd.DataFrame([new_student])
            ], ignore_index=True)
            
            st.success(f"✅ Student {name} added successfully!")
            st.balloons()

# Sidebar
st.sidebar.header("📁 Data Source")
data_option = st.sidebar.radio("Choose data source:", ["Use Sample Data", "Upload CSV", "Use My Data (Manual Entry)"])

if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    else:
        st.info("☝ Upload a CSV file to get started, or switch to another option.")
        st.stop()
elif data_option == "Use My Data (Manual Entry)":
    if len(st.session_state['user_data']) == 0:
        st.info("ℹ️ No manual data yet. Add students below, then refresh to see analysis.")
        add_student_manually()
        st.stop()
    else:
        df = st.session_state['user_data'].copy()
        st.sidebar.success(f"✅ Using your manual data ({len(df)} students)")
        
        # Show add more students option
        with st.sidebar.expander("➕ Add More Students"):
            add_student_manually()
else:
    df = generate_sample_data()
    st.sidebar.success("Sample data loaded! (500 students)")

# Show data entry form in main area if manual entry
if data_option == "Use My Data (Manual Entry)" and len(st.session_state['user_data']) > 0:
    with st.expander("➕ Add Another Student", expanded=False):
        add_student_manually()

st.markdown("---")

# Continue with the rest of the existing dashboard code
# Key Metrics
st.subheader("🎯 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

subjects = ['Mathematics', 'Science', 'English', 'History', 'Computer_Science']
if all(sub in df.columns for sub in subjects):
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        pass_rate = (df['Pass_Status'] == 'Pass').mean() * 100 if 'Pass_Status' in df.columns else 0
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col3:
        avg_marks = df['Average_Marks'].mean() if 'Average_Marks' in df.columns else 0
        st.metric("Avg Marks", f"{avg_marks:.1f}")
    with col4:
        top_subject = df[subjects].mean().idxmax()
        st.metric("Top Subject", top_subject.replace('_', ' '))
    with col5:
        topper = df.loc[df['Average_Marks'].idxmax(), 'Name'] if 'Average_Marks' in df.columns else "N/A"
        st.metric("Topper", topper)

st.markdown("---")

# Simplified Tabs
tab1, tab2 = st.tabs(["📊 Analysis", "📄 View Data"])

with tab1:
    if 'Pass_Status' in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            pass_fail_counts = df['Pass_Status'].value_counts()
            fig = px.pie(values=pass_fail_counts.values, names=pass_fail_counts.index,
                        title='Pass/Fail Distribution', color_discrete_sequence=['#2ecc71', '#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if all(sub in df.columns for sub in subjects):
                subject_avg = df[subjects].mean().sort_values(ascending=True)
                fig2 = px.bar(x=subject_avg.values, y=subject_avg.index, orientation='h',
                            title='Average Marks by Subject')
                st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("📄 Complete Data View")
    st.dataframe(df, use_container_width=True, height=400)
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="student_grades_analysis.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""<p style="text-align: center; color: gray;">
📊 Student Grades Analysis Dashboard | Created by <b>Shashwat Pathak</b>
</p>""", unsafe_allow_html=True)
