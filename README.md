# IT5006_healthcare_dataset_analysis
Repository for Data Analytics Project on University of California, Irvine, (UCI) Machine Learning Repository which contains information on patients with diabetes that represents 10 years (1999-2008) of clinical care at 130 US hospitals. 

Objective:
Hospital readmissions for diabetic patients not only impose substantial financial costs but also create significant operational strain on healthcare systems. Studies indicate that rehospitalization increases bed occupancy by approximately 10–15 %, thereby limiting capacity for new admissions and contributing to congestion within inpatient facilities (Rubin, 2018). This strain underscores the urgency for hospitals to adopt data-driven approaches that proactively identify high-risk patients before deterioration occurs.

Accordingly, this study aims to develop a predictive model to identify diabetic patients at risk of 30-day hospital readmission following their first recorded encounter. The 30-day horizon is relevant for its clinical actionability – providing a short but critical window in which targeted post-discharge interventions can prevent avoidable rehospitalizations. From an institutional standpoint, such optimization translates into measurable benefits; for example, a 500-bed hospital implementing an effective readmission-reduction strategy could save roughly USD 2 million annually through improved utilization and reduced penalty exposure (Rubin, 2018).


Declaration: GPT-AI was used for certain helper functions for data visualization or grouping of methods in this project.

Setup Guide:
1. Create a virtual environment using requirements.txt file --> if you encounter any issues with comorbidiPy library, remove it from the requirements.txt.
2. Access EDA_Preliminary_Data_Profiling.ipynb for raw notebook code for Exploratory Data Analysis
3. Under Section 4 of EDA_Preliminary_Data_Profiling, the Data Preprocessing steps are outlined and performed.
4. df_filtered_first_encounter_mapped_eng_feature.csv is the preprocessed csv output in EDA ipynb file with engineered features. 
5. An interactive exploratory data analytics dashboard can be accessed here: https://it5006-grp9.streamlit.app/
6. 3 ML models are explored for the Research Qn: Logistic Regression, Decision_Tree_Model and RandomForest. Go to the respective ipynb for review. 
