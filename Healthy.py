#pip install streamlit-option-menu

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_multi_menu import streamlit_multi_menu
import altair as alt
import base64 #This base64 module provides functions for encoding binary data to printable ASCII characters and decoding such encodings back to binary
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from plotly.subplots import make_subplots
from millify import millify # shortens values (10_000 ---> 10k)
from streamlit_extras.metric_cards import style_metric_cards
import micropip
#await micropip.install("ipywidgets")
#await micropip.install("plotly")
###
#from ipywidgets import widgets
#from sklearn.metrics import plot_confusion_matrix
#from sklearn.metrics import plot_roc_curve
#from sklearn.metrics import plot_precision_recall_curve
#from sklearn.metrics import plot_precision_recall_curve
#from sklearn.metrics import classification_report


#SET Configuration: 

#Page configuration
st.set_page_config(
    page_title="HealthCare",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded")


#STREAMLIT

import streamlit as st

#original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;">Streamlit CSS Styling✨ </h1>'
#st.markdown(original_title, unsafe_allow_html=True)


# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.unsplash.com/photo-1564352969906-8b7f46ba4b8b?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

title_writing = "Health Care"
title_format = f'<p style="text-align: center; font-family: ' \
               f'Arial; color:  #DAF7A6; font-size: 40px; ' \
               f'font-weight: bold;">{title_writing}</p>'
st.markdown(title_format, unsafe_allow_html=True)
st.markdown(background_image, unsafe_allow_html=True)

#Our Dataset
data_ = pd.read_csv("BrainTumor.csv")

#Rename Columns: 
data_.rename(columns = {'Patient ID':'Patient_ID'}, inplace = True)
data_.rename(columns = {'Tumor Type':'Tumor_Type'}, inplace = True)
data_.rename(columns = {'Tumor Grade':'Tumor_Grade'}, inplace = True)
data_.rename(columns = {'Gender':'Gender'}, inplace = True)
data_.rename(columns = {'Age':'Age'}, inplace = True)
data_.rename(columns = {'Survival Time (months)':'Survival_Time'}, inplace = True)
data_.rename(columns = {'Time to Recurrence (months)':'Time_to_Recurrence'}, inplace = True)
data_.rename(columns = {'Recurrence Site':'Recurrence_Site'}, inplace = True) 
data_.rename(columns = {'Treatment Outcome':'Treatment_Outcome'}, inplace = True)
data_.rename(columns = {'Tumor Location':'Tumor_Location'}, inplace = True)
data_.rename(columns = {'Patient ID':'Patient_ID'}, inplace = True)

# Ensure categorical columns are strings
categorical_cols = data_.select_dtypes(include=['object']).columns.tolist()
#for col in categorical_cols:
    #data_[col] = data_[col].astype(str)

#Styler 
data = data_.style.background_gradient(cmap='GnBu')
 


with st.sidebar:
    selected = option_menu("🌍 Brain Tumor Dashboard", ["Home", 'Data Explorer', 'Data Visualization', 'Predictions'], 
        icons=['house', 'male-doctor', 'directions_car', 'pill'], menu_icon="cast", default_index=1)
    selected

with st.sidebar:
    st.title('🌍 Brain Tumor Dashboard')
    
    tumor_list = list(data_.Tumor_Type.unique())  
    tumor_type = st.selectbox('Select a type', tumor_list, index=len(tumor_list)-1)
    selected_tumor_type = data_[data_.Tumor_Type == tumor_type]
    df_selected_year_sorted = selected_tumor_type.sort_values(by="Tumor_Grade", ascending=False)
    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

#First DATASET
if selected == 'Data Explorer':
    data_ = pd.read_csv("BrainTumor.csv")
    data = data_.style.background_gradient(cmap='GnBu')

#st.data_editor(data_, use_container_width=False, hide_index=None, column_order=None, column_config=None, num_rows="fixed", disabled=False, key=None, on_change=None, args=None, kwargs=None)
#Use streamlit to print out the dataset 
    st.dataframe(data)
    data_.info()
    data_.describe()
    st.header('Check Correlation')
    corr = data_.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
    correl = corr.style.background_gradient(cmap='GnBu')
    correl
    #plotly.express.scatter(data_frame=None, x=None, y=None)
    #fig = px.scatter(data_, x='Tumor Grade', y='Tumor Type')
    #fig.show()

#st.dataframe(data)
elif selected == "Data Visualization": 

    def make_heatmap(data, input_y, input_x, input_color, input_color_theme):
        heatmap = alt.Chart(data).mark_rect().encode( y=alt.Y(f'{input_y}:O', 
        axis=alt.Axis(title="Age", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)), x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)), color=alt.Color(f'max({input_color}):Q', legend=None, scale=alt.Scale(scheme=input_color_theme)),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),).properties(width=900).configure_axis(labelFontSize=12, titleFontSize=12)
        return heatmap

    data = data_
    input_x = 'Tumor_Type'
    input_y = 'Age'
    input_color_theme = selected_color_theme
    heatmap = make_heatmap(data, input_y, input_x, 'Time_to_Recurrence', input_color_theme)
    heatmap

    ###############################

    st.header("Time to Recurrence (months), Grouped by Tumor Type")
    recurrence_by_type_avg = data_.groupby('Tumor_Type')['Time_to_Recurrence'].mean().round()
    recurrence_by_type_avg = recurrence_by_type_avg.reset_index()
    Tumor_Type_ = recurrence_by_type_avg['Tumor_Type']
    Average_recurrence = recurrence_by_type_avg['Time_to_Recurrence']

    # Define custom colors for each tumor type
    color_map = {
    'Astrocytoma': 'aliceblue', 
    'Glioblastoma': 'aquamarine', 
    'Meningioma': 'darkcyan' }
    # Plotly bar chart
    fig_1 = px.bar(recurrence_by_type_avg, x= Tumor_Type_, y= Average_recurrence, 
                color='Tumor_Type', title='Average Time to Recurrence (months) Grouped by Tumor Type', color_discrete_map= color_map)
    # Customize layout 
    fig_1.update_layout(xaxis_title='Tumor Type', yaxis_title='Average Time to Recurrence (months)',
                    plot_bgcolor='rgba(0,0,0,0)')  # transparent background    
    
    st.plotly_chart(fig_1)

    ###################################

    # this function get the % change for any column by year and the specified aggregate
    def get_per_age_change(col,data_,metric):
        recurrence_by_type_avg = data_.groupby('Tumor_Type')['Time_to_Recurrence'].mean().round()
        recurrence_by_type_avg = recurrence_by_type_avg.reset_index()
        Tumor_Type_ = recurrence_by_type_avg['Tumor_Type']
        Average_recurrence = recurrence_by_type_avg['Time_to_Recurrence']

    # creates the container for page title
    dash_1 = st.container()

    with dash_1:
        st.markdown("<h2 style='text-align: center; color: darkcyan;'>Brain Tumor Dashboard</h2>", unsafe_allow_html=True)
        #st.write("")

    # creates the container for metric card
    dash_2 = st.container()

    with dash_2:
        # get kpi metrics
        Tumor_Type_ = data_['Tumor_Type'].nunique()
        Tumor_Grade_ = data_['Tumor_Grade'].nunique()
        Age = data_['Age'].nunique()

        col1, col2, col3 = st.columns(3)
        # create column span
        col1.metric(label="Tumor Type", value= " "+millify(Tumor_Type_, precision=2)) #,delta=sales_per_change
        
        col2.metric(label="Tumor Grade", value= " "+millify(Tumor_Grade_, precision=2)) #, delta=profit_per_change
        
        col3.metric(label="Age Unique", value=Age) #, delta=order_count_per_change
        
        # this is used to style the metric card
        style_metric_cards(border_left_color="darkcyan")
        #st.markdown(styly("<h2 style='color: black;'>Your Metric Title</h2>", border_left_color="darkcyan"), unsafe_allow_html=True)

        # container for top 10 best selling and most profitable products
    
    dash_3 = st.container()
    with dash_3:
        
        # create columna for both graph
        col1,col2 = st.columns(2)
        # get the top 10 best selling products
        Treatment_out = data_.groupby('Tumor_Grade')['Time_to_Recurrence'].mean().round()
        Treatment_out = pd.DataFrame(Treatment_out).reset_index()

        # get the top 10 most profitable products
        Treat_Type = data_.groupby('Tumor_Type')['Time_to_Recurrence'].sum()
        Treat_Type = pd.DataFrame(Treat_Type).reset_index()
    
        # create the altair chart
        import plotly.io as pio

        #fig_3 = go.Figure(data=...)
        #html_content = pio.to_html(fig, include_plotlyjs='cdn')
        # Define custom colors for each tumor type
        color_map = {
        'Astrocytoma': 'green', 
        'Glioblastoma': 'blue', 
        'Meningioma': 'dark' }
        with col1:
            fig_3 = px.scatter(data_, x="Time_to_Recurrence", y="Age",
                                size="Age", color_discrete_map={
        'Astrocytoma': 'red',
        'Glioblastoma': 'green'
    },
                                    hover_name="Tumor_Type", log_x=True, size_max=20, width=610, height=350)
            html_content = pio.to_html(fig_3, include_plotlyjs='cdn')
            fig_3.show()
            fig_3.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',})
            st.plotly_chart(fig_3)

        with col2 : 
            fig_4 = px.bar(data_, x="Age", y="Survival_Time", color="Tumor_Location", title="Long-Form Input", width=610, height=350)
            fig_4.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',})
            fig_4.show()
            st.plotly_chart(fig_4)

            

        # create the altair chart
        #with col2:
            #chart = alt.Chart(Treat_Type).mark_bar(opacity=0.9,color="#9FC131").encode(
                    #x=''sum(Profit):Q'',
                    #y=alt.Y(''Product Name:N'', sort=''-x'')
                    
                #)
            #chart = chart.properties(title="Top 10 Most Profitable Products" )

            #st.altair_chart(chart,use_container_width=True)


# Display the encoded DataFrame
#original_data
# Predictions section

elif selected == 'Predictions':

    ####################################
    #MHANDLE MISSING VALUES 
    ####################################

    # Handle missing values
    missing_v = {'Recurrence_Site': 'Full Recovery', 'Time_to_Recurrence': 0}
    data_.fillna(missing_v, inplace=True)

    ####################################
    #HANDLE OUTLIERS
    ####################################

    #Using IQR
    #Read the dataset
    data = pd.read_csv("BrainTumor.csv")
    #IQR
    Q1 = np.percentile(data['Survival Time (months)'], 25)
    Q3 = np.percentile(data['Survival Time (months)'], 75)
    IQR = Q3 - Q1
    threshold = 1.5
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    # Identify outliers
    outliers = (data['Survival Time (months)'] < lower_bound) | (data['Survival Time (months)'] > upper_bound)
    # Remove outliers
    data_cleaned = data[~outliers]
    data = data_cleaned

    ####################################
    #GET OUR ENCODED VALUES
    ####################################

    # Create a dictionary to store mappings
    encoding_mappings = {}

    # Iterate trough columns
    for col in data_.columns:
        if data_[col].dtype == 'object':  # Verify if our columns type is object
            le = LabelEncoder()
            data_[col] = le.fit_transform(data_[col].astype(str))
            # Store the mapping in the dictionary
            encoding_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    corresponding = pd.DataFrame(encoding_mappings)
    # Replace NaN values with '__'
    corresponding.fillna('__', inplace=True)
    corresponding_ = corresponding.style.background_gradient(cmap='GnBu')
    st.dataframe(corresponding_, 1000, 300)

    #Print out our dataframe
    #original_data_

    
    ####################################
    #MAKE PREDICTIONS
    ####################################
    #extract x and y from our data
    x = data_.drop(columns=['Treatment_Outcome']).values
    y = data_['Treatment_Outcome'].values
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=40) #splitting data with test size of 35%
    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train the classifier
    rf_classifier.fit(X_train, y_train)
    # Predict on the test set
    predicted = rf_classifier.predict(X_test)

    if st.button('Predict Treatment Outcome'):

        # Calculate additional metrics
        precision = precision_score(y_test, predicted, average='weighted') #, average='macro' 
        recall = recall_score(y_test, predicted, average= 'weighted') # , average='macro'
        f1 = f1_score(y_test, predicted, average= 'weighted') #, average='macro'

        #InvalidParameterError: The 'average' parameter of precision_score must be a str among 
        # {'macro', 'micro', 'weighted', 'samples', 'binary'} or None. Got 'None' instead.
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predicted)
        st.write("Check Our Accuracy: ", accuracy) 
        # Display metrics
        st.write("Precision: ", precision)
        st.write("Recall: ", recall)
        st.write("F1 Score: ", f1)

        # Display classification report
        st.header(":green[_Classification Report_]")
        report_df = pd.DataFrame(classification_report(y_test, predicted, output_dict=True)).transpose()
        report_df = report_df.style.background_gradient(cmap='GnBu')
        st.dataframe(report_df, 1000, 300)

        # Display confusion matrix
        st.header(":green[_Confusion Matrix_]")
        conf_matrix = confusion_matrix(y_test, predicted)
        conf_matrix_df = pd.DataFrame(conf_matrix,
                                      index=[f'Actual {cls}' for cls in np.unique(y)],
                                      columns=[f'Predicted {cls}' for cls in np.unique(y)])
        conf_matrix_df = conf_matrix_df.style.background_gradient(cmap='GnBu')
        st.dataframe(conf_matrix_df, 1000, 300)

        # Display feature importancegreen
        st.header(":green[_Feature Importances_]")
        importances = rf_classifier.feature_importances_
        feature_importances_df = pd.DataFrame({'Feature': data_.drop(columns=['Treatment_Outcome']).columns,'Importance': importances}).sort_values(by='Importance', ascending=False)
        feature_importances_df = feature_importances_df.style.background_gradient(cmap='GnBu')
        st.dataframe(feature_importances_df, 1000, 300)

        ####################################
        # DISPLAY ROC CURVE AND AUC
        ####################################
        st.header("Show ROC Curve and AUC") 
        predicted_proba = rf_classifier.predict_proba(X_test)

        # Binarize the output
        y_bin = label_binarize(y_test, classes=np.unique(y)) #Why do we need to binarize ? 
        n_classes = y_bin.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], predicted_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve with Plotly
        roc_traces = []
        colors = cycle(['#FF5733', '#33FF57', '#3357FF'])  # Example colors
        for i, color in zip(range(n_classes), colors):
            trace = go.Scatter(
                x=fpr[i], y=tpr[i],
                mode='lines',
                name=f'Class {i} (AUC = {roc_auc[i]:.2f})',
                line=dict(color=color, width=2)
            )
            roc_traces.append(trace)

        # Diagonal line
        diagonal_line = go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Chance',
            line=dict(color='black', dash='dash')
        )
        roc_traces.append(diagonal_line)

        # Layout
        roc_layout = go.Layout(
            title='Receiver Operating Characteristic (ROC)',
            xaxis=dict(title='False Positive Rate'),
            yaxis=dict(title='True Positive Rate'),
            showlegend=True
        )

        # Create figure
        roc_fig = go.Figure(data=roc_traces, layout=roc_layout)

        # Display with Streamlit
        st.plotly_chart(roc_fig)
        
        
        
        
        #predicted_proba = rf_classifier.predict_proba(X_test)  # Get probabilities for all classes
        # Binarize the output
        #y_bin = label_binarize(y_test, classes=np.unique(y))
        #n_classes = y_bin.shape[1]

        # Compute ROC curve and ROC area for each class
        #fpr = dict()
        #tpr = dict()
        #roc_auc = dict()
        #for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], predicted_proba[:, i])
            #roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        #fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), predicted_proba.ravel())
        #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curve for each class
        #plt.figure(figsize=(10, 6))
        #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        #for i, color in zip(range(n_classes), colors):
            #plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        #label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

        #plt.plot([0, 1], [0, 1], 'k--', lw=2)
        #plt.xlim([0.0, 1.0])
        #plt.ylim([0.0, 1.05])
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title('Receiver Operating Characteristic (ROC)')
        #plt.legend(loc="lower right")
        #fig1.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
        #st.pyplot(plt)


        # Calculating metrics
        #Precision = precision_score(y_test, predicted, average='macro')
        #Recall = recall_score(y_test, predicted, average='macro')
        #F1_Score =  f1_score(y_test, predicted, average='macro')
        #Report = classification_report(y_test, predicted, output_dict=True)
        #Confusion_Matrix = confusion_matrix(y_test, predicted)

        # Display Classification Report
        #st.header("Classification Report")
        #report_df = pd.DataFrame(Report).transpose()
        #st.dataframe(report_df)

        # Display Confusion Matrix
        #st.header("Confusion Matrix")
        #conf_matrix = pd.DataFrame(Confusion_Matrix, 
                                    #index=[f'Actual {cls}' for cls in data_.target_names], 
                                    #columns=[f'Predicted {cls}' for cls in data_.target_names])
        #st.dataframe(Confusion_Matrix)

        #importances = rf_classifier.feature_importances_
        #feature_importances_df = pd.DataFrame({'Feature': x.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        #st.dataframe(feature_importances_df)
        #st.dataframe(importances)
