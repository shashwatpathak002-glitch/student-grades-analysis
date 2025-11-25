import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Page Configuration
st.set_page_config(
    page_title="Student Grades Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Title
st.markdown('<h1 class="main-header">📊 Student Grades Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: gray;">Created by Shashwat Pathak</p>', unsafe_allow_html=True)
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

# Sidebar
st.sidebar.header("📁 Data Source")
data_option = st.sidebar.radio("Choose data source:", ["Use Sample Data", "Upload Your Own CSV"])

if data_option == "Upload Your Own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    else:
        st.info("☝ Upload a CSV file to get started, or switch to Sample Data.")
        st.stop()
else:
    df = generate_sample_data()
    st.sidebar.success("Sample data loaded! (500 students)")

# Data Cleaning Section
st.sidebar.markdown("---")
st.sidebar.header("🧹 Data Cleaning")

# Handle missing values
if df.isnull().sum().sum() > 0:
    st.sidebar.warning(f"Found {df.isnull().sum().sum()} missing values")
    if st.sidebar.button("Clean Missing Values"):
        df = df.fillna(df.median(numeric_only=True))
        st.sidebar.success("Missing values cleaned!")
else:
    st.sidebar.success("✓ No missing values found")

# Main Dashboard
# Key Metrics
st.subheader("🎯 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

subjects = ['Mathematics', 'Science', 'English', 'History', 'Computer_Science']
if all(sub in df.columns for sub in subjects):
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        pass_rate = (df['Pass_Status'] == 'Pass').mean() * 100 if 'Pass_Status' in df.columns else (df['Average_Marks'] >= 40).mean() * 100
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col3:
        avg_marks = df['Average_Marks'].mean() if 'Average_Marks' in df.columns else df[subjects].mean().mean()
        st.metric("Avg Marks", f"{avg_marks:.1f}")
    with col4:
        top_subject = df[subjects].mean().idxmax()
        st.metric("Top Subject", top_subject.replace('_', ' '))
    with col5:
        topper = df.loc[df['Average_Marks'].idxmax(), 'Name'] if 'Average_Marks' in df.columns else "N/A"
        st.metric("Topper", topper)
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Columns", len(df.columns))

st.markdown("---")

# Tabs for Analysis
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Pass/Fail Analysis", "📚 Subject Analysis", "🔗 Correlation Analysis", "🏆 Top Students", "📄 Data View"])

with tab1:
    st.subheader("Pass/Fail Ratio Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Pass_Status' in df.columns:
            pass_fail_counts = df['Pass_Status'].value_counts()
            fig_pie = px.pie(values=pass_fail_counts.values, names=pass_fail_counts.index,
                           title='Pass/Fail Distribution', color_discrete_sequence=['#2ecc71', '#e74c3c'])
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        if 'Grade' in df.columns:
            grade_counts = df['Grade'].value_counts().sort_index()
            fig_bar = px.bar(x=grade_counts.index, y=grade_counts.values,
                           title='Grade Distribution', labels={'x': 'Grade', 'y': 'Count'},
                           color=grade_counts.values, color_continuous_scale='viridis')
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Gender-wise pass rate
    if 'Gender' in df.columns and 'Pass_Status' in df.columns:
        st.subheader("Gender-wise Pass Rate")
        gender_pass = df.groupby('Gender')['Pass_Status'].apply(lambda x: (x == 'Pass').mean() * 100).reset_index()
        gender_pass.columns = ['Gender', 'Pass_Rate']
        fig_gender = px.bar(gender_pass, x='Gender', y='Pass_Rate', title='Pass Rate by Gender',
                          color='Gender', text='Pass_Rate')
        fig_gender.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_gender, use_container_width=True)

with tab2:
    st.subheader("Subject-wise Analysis")
    
    if all(sub in df.columns for sub in subjects):
        # Average marks by subject
        subject_avg = df[subjects].mean().sort_values(ascending=True)
        fig_subject = px.bar(x=subject_avg.values, y=subject_avg.index, orientation='h',
                           title='Average Marks by Subject', labels={'x': 'Average Marks', 'y': 'Subject'},
                           color=subject_avg.values, color_continuous_scale='blues')
        st.plotly_chart(fig_subject, use_container_width=True)
        
        # Box plot for subjects
        st.subheader("Marks Distribution by Subject")
        df_melted = df[subjects].melt(var_name='Subject', value_name='Marks')
        fig_box = px.box(df_melted, x='Subject', y='Marks', title='Subject-wise Marks Distribution',
                        color='Subject')
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Subject difficulty analysis
        st.subheader("Subject Difficulty Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fail_rate_by_subject = {sub: (df[sub] < 40).mean() * 100 for sub in subjects}
            fail_df = pd.DataFrame(list(fail_rate_by_subject.items()), columns=['Subject', 'Fail_Rate'])
            fig_fail = px.bar(fail_df, x='Subject', y='Fail_Rate', title='Fail Rate by Subject (%)',
                            color='Fail_Rate', color_continuous_scale='reds')
            st.plotly_chart(fig_fail, use_container_width=True)
        with col2:
            std_by_subject = df[subjects].std().sort_values(ascending=False)
            fig_std = px.bar(x=std_by_subject.index, y=std_by_subject.values,
                           title='Standard Deviation by Subject', labels={'x': 'Subject', 'y': 'Std Dev'})
            st.plotly_chart(fig_std, use_container_width=True)

with tab3:
    st.subheader("🔗 Correlation Analysis - Key Factors Affecting Marks")
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Correlation heatmap
        corr_matrix = df[numeric_cols].corr()
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig_heatmap.update_layout(title='Correlation Heatmap - Factors Affecting Performance', height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Key insights
        st.subheader("💡 Key Insights - Factors Affecting Marks")
        
        if 'Average_Marks' in df.columns:
            correlations_with_avg = corr_matrix['Average_Marks'].drop('Average_Marks').sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Positive Correlations (Higher = Better Marks):**")
                positive_corr = correlations_with_avg[correlations_with_avg > 0.1]
                for factor, corr in positive_corr.items():
                    st.write(f"✅ {factor}: {corr:.3f}")
            with col2:
                st.markdown("**Negative Correlations:**")
                negative_corr = correlations_with_avg[correlations_with_avg < -0.1]
                for factor, corr in negative_corr.items():
                    st.write(f"❌ {factor}: {corr:.3f}")
        
        # Scatter plots for key factors
        st.subheader("Factor Impact Visualization")
        if 'Study_Hours_Per_Day' in df.columns and 'Average_Marks' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig_scatter1 = px.scatter(df, x='Study_Hours_Per_Day', y='Average_Marks',
                                        title='Study Hours vs Average Marks', trendline='ols',
                                        color='Pass_Status' if 'Pass_Status' in df.columns else None)
                st.plotly_chart(fig_scatter1, use_container_width=True)
            with col2:
                if 'Attendance_Percentage' in df.columns:
                    fig_scatter2 = px.scatter(df, x='Attendance_Percentage', y='Average_Marks',
                                            title='Attendance vs Average Marks', trendline='ols',
                                            color='Pass_Status' if 'Pass_Status' in df.columns else None)
                    st.plotly_chart(fig_scatter2, use_container_width=True)

with tab4:
    st.subheader("🏆 Top Performing Students")
    
    if 'Average_Marks' in df.columns:
        n_top = st.slider("Select number of top students to display:", 5, 50, 10)
        
        top_students = df.nlargest(n_top, 'Average_Marks')[['Student_ID', 'Name', 'Average_Marks', 'Grade', 'Pass_Status'] + subjects]
        
        st.dataframe(top_students.style.background_gradient(subset=['Average_Marks'], cmap='Greens'), use_container_width=True)
        
        # Performance by parent education
        if 'Parent_Education' in df.columns:
            st.subheader("Performance by Parent Education Level")
            parent_perf = df.groupby('Parent_Education')['Average_Marks'].mean().sort_values(ascending=True)
            fig_parent = px.bar(x=parent_perf.values, y=parent_perf.index, orientation='h',
                              title='Average Marks by Parent Education', labels={'x': 'Average Marks', 'y': 'Parent Education'},
                              color=parent_perf.values, color_continuous_scale='viridis')
            st.plotly_chart(fig_parent, use_container_width=True)
        
        # Impact of extra classes
        if 'Extra_Classes' in df.columns:
            st.subheader("Impact of Extra Classes")
            extra_class_impact = df.groupby('Extra_Classes')['Average_Marks'].mean()
            fig_extra = px.bar(x=extra_class_impact.index, y=extra_class_impact.values,
                             title='Average Marks: Extra Classes vs No Extra Classes',
                             labels={'x': 'Extra Classes', 'y': 'Average Marks'},
                             color=extra_class_impact.values, color_continuous_scale='blues')
            st.plotly_chart(fig_extra, use_container_width=True)

with tab5:
    st.subheader("📄 Complete Data View")
    
    # Data statistics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset Shape:**")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    with col2:
        st.markdown("**Data Types:**")
        st.write(df.dtypes.value_counts())
    
    # Display full data
    st.dataframe(df, use_container_width=True, height=400)
    
    # Download options
    st.subheader("📥 Download Data")
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="student_grades_analysis.csv",
            mime="text/csv"
        )
    with col2:
        # Summary statistics
        summary = df.describe()
        summary_csv = summary.to_csv()
        st.download_button(
            label="Download Summary Statistics",
            data=summary_csv,
            file_name="summary_statistics.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""<p style="text-align: center; color: gray;">
📊 Student Grades Analysis Dashboard | Created by <b>Shashwat Pathak</b> | 
<a href="https://github.com/shashwatpathak002-glitch" target="_blank">GitHub</a>
</p>""", unsafe_allow_html=True)
