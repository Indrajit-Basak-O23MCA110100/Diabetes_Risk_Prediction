import streamlit as st 
import joblib
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import cross_val_score, KFold
import plotly.graph_objs as go

attrib_info = """
#### Attribute Information:
    - Age 1.20-65
    - Sex 1. Male, 2.Female
    - Polyuria 1.Yes, 2.No.
    - Polydipsia 1.Yes, 2.No.
    - Sudden Weight Loss 1.Yes, 2.No.
    - Weakness 1.Yes, 2.No.
    - Polyphagia 1.Yes, 2.No.
    - Genital Thrush 1.Yes, 2.No.
    - Visual Blurring 1.Yes, 2.No.
    - Itching 1.Yes, 2.No.
    - Irritability 1.Yes, 2.No.
    - Delayed Healing 1.Yes, 2.No.
    - Partial Paresis 1.Yes, 2.No.
    - Muscle Stiffness 1.Yes, 2.No.
    - Alopecia 1.Yes, 2.No.
    - Obesity 1.Yes, 2.No.
    - Class 1.Positive, 2.Negative.

"""
label_dict = {"No":0,"Yes":1}
gender_map = {"Female":0,"Male":1}
target_label_map = {"Negative":0,"Positive":1}

df_clean=pd.read_csv("data/diabetes_data_clean.csv")

# Plot feature importance
def get_features_importances(title,features,features_importances):
    trace = go.Scatter(y = features_importances, x = features, mode='markers',
    	marker=dict(sizemode = 'diameter', sizeref = 1, size = 25, color = features_importances,
            colorscale='Portland', showscale=True),
        text = features)
    data = [trace]
    layout= go.Layout(autosize= True,title= title,hovermode= 'closest',width=600,
        yaxis=dict(title= 'Feature Importance',ticklen= 5,gridwidth= 2),
		showlegend= False)
    fig = go.Figure(data=data, layout=layout)
    return fig

def get_fvalue(val):
	feature_dict = {"No":0,"Yes":1}
	for key,value in feature_dict.items():
		if val == key:
			return value 

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 



# Load ML Models
@st.cache_resource
def load_model(model_path):
	with open(model_path, "rb") as file:
		loaded_model = joblib.load(file)
	return loaded_model


def run_ml_app():
	st.subheader("Machine Learning Section")

	submenu = st.sidebar.selectbox("Sub Menu",["Model Prediction","Model Comparison"])

    ################ Go to Model Prediction subsection ###########################
	if submenu == "Model Prediction":
		st.subheader("Model Prediction")
		with st.expander("Attributes Info"):
			st.markdown(attrib_info,unsafe_allow_html=True)
	
        ### Users need to select the model according to their choice 
		model_name = st.selectbox("Select Machine Learning Model", ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier"])

		# Layout
		col1,col2 = st.columns(2)

        ### UI elements for input parameters
		with col1:
			age = st.number_input("Age",10,100)
			gender = st.radio("Gender",("Female","Male"))
			polyuria = st.radio("Polyuria",["No","Yes"])
			polydipsia = st.radio("Polydipsia",["No","Yes"]) 
			sudden_weight_loss = st.selectbox("Sudden Weight Loss",["No","Yes"])
			weakness = st.radio("Weakness",["No","Yes"]) 
			polyphagia = st.radio("Polyphagia",["No","Yes"]) 
			genital_thrush = st.selectbox("Genital Thrush",["No","Yes"]) 
		
	
		with col2:
			visual_blurring = st.selectbox("Visual Blurring",["No","Yes"])
			itching = st.radio("Itching",["No","Yes"]) 
			irritability = st.radio("Irritability",["No","Yes"]) 
			delayed_healing = st.radio("Delayed Healing",["No","Yes"]) 
			partial_paresis = st.selectbox("Partial Paresis",["No","Yes"])
			muscle_stiffness = st.radio("Muscle Stiffness",["No","Yes"]) 
			alopecia = st.radio("Alopecia",["No","Yes"]) 
			obesity = st.select_slider("Obesity",["No","Yes"]) 

        # Display the inputs given by the user
		with st.expander("Your Selected Options"):
			result = {
			'selected_machine_learning_model': model_name,
			'age':age,
			'gender':gender,
			'polyuria':polyuria,
			'polydipsia':polydipsia,
			'sudden_weight_loss':sudden_weight_loss,
			'weakness':weakness,
			'polyphagia':polyphagia,
			'genital_thrush':genital_thrush,
			'visual_blurring':visual_blurring,
			'itching':itching,
			'irritability':irritability,
			'delayed_healing':delayed_healing,
			'partial_paresis':partial_paresis,
			'muscle_stiffness':muscle_stiffness,
			'alopecia':alopecia,
			'obesity':obesity}
			st.write(result)
			encoded_result = []
			for i in result.values():
				if type(i) == int:
					encoded_result.append(i)
				elif i in ["Female","Male"]:
					res = get_value(i,gender_map)
					encoded_result.append(res)
				elif i in ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier"]:
					pass
				else:
					encoded_result.append(get_fvalue(i))

        # Display prediction result
		with st.expander("Prediction Results"):
			single_sample = np.array(encoded_result).reshape(1,-1)

			### Fetch the selected model
			selected_model = result["selected_machine_learning_model"]

            ### Load the model that is selected by the user
			
            ### If the user selects Logistic Regression
			if selected_model=="Logistic Regression":
				loaded_model = load_model("models/logistic_regression_model_grid_search_new.pkl")
				model_markdown = '<p style="font-size:20px;">Prediction Made Using Logistic Regression Model</p>'
				st.markdown(model_markdown,unsafe_allow_html=True)

            ### If the user selects Decision Tree Classifier
			elif selected_model=="Decision Tree Classifier":
				loaded_model = load_model("models/decision_tree_model_depth_5.pkl")
				model_markdown = '<p style="font-size:20px;">Prediction Made Using Decision Tree Classifier Model</p>'
				st.markdown(model_markdown,unsafe_allow_html=True)

            ### If the user selects Random Forest Classifier
			elif selected_model=="Random Forest Classifier":
				loaded_model = load_model("models/random_forest_model.pkl")
				model_markdown = '<p style="font-size:20px;">Prediction Made Using Random Forest Classifier Model</p>'
				st.markdown(model_markdown,unsafe_allow_html=True)

            ### If the user selects Support Vector Classifier
			elif selected_model=="Support Vector Classifier":
				loaded_model = load_model("models/svc_grid_model_final.pkl")
				model_markdown = '<p style="font-size:20px;">Prediction Made Using Support Vector Classifier Model</p>'
				st.markdown(model_markdown,unsafe_allow_html=True)

            ### Display the predicted class label
			markdown = '<p style="font-size:17px;">Predited Class Value</p>'
			st.markdown(markdown,unsafe_allow_html=True)

			prediction = loaded_model.predict(single_sample)
			pred_prob = loaded_model.predict_proba(single_sample)
			pred_probability_score = {"Negative Risk":pred_prob[0][0]*100,"Positive Risk":pred_prob[0][1]*100}

			if(selected_model=="Support Vector Classifier"):
				if(pred_probability_score["Negative Risk"] > pred_probability_score["Positive Risk"]):
					prediction = np.array([0])
					st.write(prediction)
				else:
					prediction=np.array([1])
					st.write(prediction)
			else:
				st.write(prediction)	

            ### Display the prediction probability score 
			if(pred_probability_score["Negative Risk"] > pred_probability_score["Positive Risk"]):
				st.success("Negative Risk: {:.2f} %".format(pred_probability_score["Negative Risk"]))
			else:
				st.warning("Positive Risk: {:.2f} %".format(pred_probability_score["Positive Risk"]))
		
			st.subheader("Prediction Probability Score")
			st.json(pred_probability_score)

    ################ Go to Model Comparison subsection ###########################
	else:
		st.subheader("Model Comparison")
		
        ### Users need to select the model according to their choice 
		model_name = st.selectbox("Select Machine Learning Model", ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier"])
			
        ### If the user selects Logistic Regression
		if model_name == "Logistic Regression":
			## Display classification report
			with st.expander("Classification Report"):
				clf_report_df=pd.read_csv("model_evaluation/logistic_regression/lr_clf_report.csv")
				clf_report_df.rename(columns={'Unnamed: 0': 'Category'}, inplace=True)
				clf_report_df.set_index('Category', inplace=True)
				st.dataframe(clf_report_df)
			
            ## Display cross validation accuracy
			with st.expander("Cross Validation Accuracy"):
				model = load_model("models/logistic_regression_model_grid_search_new.pkl")
				# Get the index of the best estimator
				best_index = model.best_index_

				# Extract scores for the best estimator (5 fold cross validation)
				best_cv_scores = model.cv_results_["split0_test_score"][best_index], \
                 model.cv_results_["split1_test_score"][best_index], \
                 model.cv_results_["split2_test_score"][best_index], \
                 model.cv_results_["split3_test_score"][best_index], \
                 model.cv_results_["split4_test_score"][best_index]

				best_cv_scores = [float(score) for score in best_cv_scores]

				score_dict={
					"Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
					"Accuracy Score": best_cv_scores
					}
				
				score_dict['Accuracy Score'] = [round(score, 3) for score in score_dict['Accuracy Score'][:len(score_dict['Fold'])]]
				st.dataframe(score_dict)

				mean_test_score = model.cv_results_["mean_test_score"][best_index]
				std_test_score = model.cv_results_["std_test_score"][best_index]  

				st.write(f"**Mean Accuracy: {mean_test_score:.3f}**")
				st.write(f"**Standard Deviation: {std_test_score:.3f}**")

				st.success("**Higher mean score is desireable as it indicates that the model performs well across multiple validation sets.**")
				st.success("**Lower the standard deviation, better the model is. It indicates how consistent or stable the model's performance is across different data splits.**")

            ## Display confusion matrix
			with st.expander("Confusion Matrix"):
				conf_matrix = Image.open("model_evaluation/logistic_regression/lr_confusion_matrix.png")
				st.image(conf_matrix, use_container_width=False)
			
            ## Display ROC curve
			with st.expander("Receiver Operating Characteristic (ROC) Curve"):
				roc_curve = Image.open("model_evaluation/logistic_regression/lr_roc_curve.png")
				st.image(roc_curve, use_container_width=False)
				st.success("**Higher the AUC, better the ability of the model to separate the two classes, regardless of the threshold.**")

        ### If the user selects Decision Tree Classifier
		elif model_name == "Decision Tree Classifier":
			## Display feature importance
			with st.expander("Feature Importance"):
				model = load_model("models/decision_tree_model_depth_5.pkl")
				X = df_clean.columns[:-1]
				plot=get_features_importances("Decision Tree Feature Importance", X,
						   model.feature_importances_)
				st.plotly_chart(plot, use_container_width=False)

            ## Display classification report
			with st.expander("Classification Report"):
				clf_report_df=pd.read_csv("model_evaluation/decision_tree/dt_clf_report.csv")
				clf_report_df.rename(columns={'Unnamed: 0': 'Category'}, inplace=True)
				clf_report_df.set_index('Category', inplace=True)
				st.dataframe(clf_report_df)
			
            ## Display cross validation accuracy
			with st.expander("Cross Validation Accuracy"):
				model = load_model("models/decision_tree_model_depth_5.pkl")
				X = df_clean.drop(columns=['class'])
				y=df_clean['class']

				# K-Fold Cross Validation (K=5)
				kfold = KFold(n_splits=5, shuffle=True, random_state=42)
				scores = cross_val_score(model, X, y, scoring='accuracy', cv=kfold)
			
				score_dict={
					"Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
					"Accuracy Score": scores
					}
				score_dict["Accuracy Score"] = score_dict["Accuracy Score"].tolist()
				score_dict['Accuracy Score'] = [round(score, 3) for score in score_dict['Accuracy Score'][:len(score_dict['Fold'])]]
				st.dataframe(score_dict)

				st.write(f"**Mean Accuracy: {scores.mean():.3f}**")
				st.write(f"**Standard Deviation: {scores.std():.3f}**")

				st.success("**Higher mean score is desireable as it indicates that the model performs well across multiple validation sets.**")
				st.success("**Lower the standard deviation, better the model is. It indicates how consistent or stable the model's performance is across different data splits.**")

            ## Display confusion matrix
			with st.expander("Confusion Matrix"):
				conf_matrix = Image.open("model_evaluation/decision_tree/dt_confusion_matrix.png")
				st.image(conf_matrix, use_container_width=False)
			
            ## Display ROC curve
			with st.expander("Receiver Operating Characteristic (ROC) Curve"):
				roc_curve = Image.open("model_evaluation/decision_tree/dt_roc_curve.png")
				st.image(roc_curve, use_container_width=False)
				st.success("**Higher the AUC, better the ability of the model to separate the two classes, regardless of the threshold.**")
		
        ### If the user selects Random Forest Classifier
		elif model_name == "Random Forest Classifier":
			## Display feature importance
			with st.expander("Feature Importance"):
				model = load_model("models/random_forest_model.pkl")
				X = df_clean.columns[:-1]
				plot=get_features_importances("Random Forest Feature Importance", X,
						   model.feature_importances_)
				st.plotly_chart(plot, use_container_width=False)

            ## Display classification report
			with st.expander("Classification Report"):
				clf_report_df=pd.read_csv("model_evaluation/random_forest/rf_clf_report.csv")
				clf_report_df.rename(columns={'Unnamed: 0': 'Category'}, inplace=True)
				clf_report_df.set_index('Category', inplace=True)
				st.dataframe(clf_report_df)
			
            ## Display cross validation accuracy
			with st.expander("Cross Validation Accuracy"):
				model = load_model("models/random_forest_model.pkl")
				X = df_clean.drop(columns=['class'])
				y=df_clean['class']

				# K-Fold Cross Validation (K=5)
				kfold = KFold(n_splits=5, shuffle=True, random_state=42)
				scores = cross_val_score(model, X, y, scoring='accuracy', cv=kfold)
			
				score_dict={
					"Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
					"Accuracy Score": scores
					}
				score_dict["Accuracy Score"] = score_dict["Accuracy Score"].tolist()
				score_dict['Accuracy Score'] = [round(score, 3) for score in score_dict['Accuracy Score'][:len(score_dict['Fold'])]]
				st.dataframe(score_dict)

				st.write(f"**Mean Accuracy: {scores.mean():.3f}**")
				st.write(f"**Standard Deviation: {scores.std():.3f}**")

				st.success("**Higher mean score is desireable as it indicates that the model performs well across multiple validation sets.**")
				st.success("**Lower the standard deviation, better the model is. It indicates how consistent or stable the model's performance is across different data splits.**")

			## Display confusion matrix
			with st.expander("Confusion Matrix"):
				conf_matrix = Image.open("model_evaluation/random_forest/rf_confusion_matrix.png")
				st.image(conf_matrix, use_container_width=False)
			
            ## Display ROC curve
			with st.expander("Receiver Operating Characteristic (ROC) Curve"):
				roc_curve = Image.open("model_evaluation/random_forest/rf_roc_curve.png")
				st.image(roc_curve, use_container_width=False)
				st.success("**Higher the AUC, better the ability of the model to separate the two classes, regardless of the threshold.**")

        ### If the user selects Support Vector Classifier
		elif model_name == "Support Vector Classifier":
			## Display classification report
			with st.expander("Classification Report"):
				clf_report_df=pd.read_csv("model_evaluation/support_vector_classifier/svc_clf_report.csv")
				clf_report_df.rename(columns={'Unnamed: 0': 'Category'}, inplace=True)
				clf_report_df.set_index('Category', inplace=True)
				st.dataframe(clf_report_df)

            ## Display cross validation accuracy
			with st.expander("Cross Validation Accuracy"):
				model = load_model("models/svc_grid_model_final.pkl")
				# Get the index of the best estimator
				best_index = model.best_index_

				# Extract scores for the best estimator (5 fold cross validation)
				best_cv_scores = model.cv_results_["split0_test_score"][best_index], \
                 model.cv_results_["split1_test_score"][best_index], \
                 model.cv_results_["split2_test_score"][best_index], \
                 model.cv_results_["split3_test_score"][best_index], \
                 model.cv_results_["split4_test_score"][best_index]

				best_cv_scores = [float(score) for score in best_cv_scores]

				score_dict={
					"Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
					"Accuracy Score": best_cv_scores
					}
				
				score_dict['Accuracy Score'] = [round(score, 3) for score in score_dict['Accuracy Score'][:len(score_dict['Fold'])]]
				st.dataframe(score_dict)

				mean_test_score = model.cv_results_["mean_test_score"][best_index]
				std_test_score = model.cv_results_["std_test_score"][best_index]  

				st.write(f"**Mean Accuracy: {mean_test_score:.3f}**")
				st.write(f"**Standard Deviation: {std_test_score:.3f}**")

				st.success("**Higher mean score is desireable as it indicates that the model performs well across multiple validation sets.**")
				st.success("**Lower the standard deviation, better the model is. It indicates how consistent or stable the model's performance is across different data splits.**")
		
            ## Display confusion matrix
			with st.expander("Confusion Matrix"):
				conf_matrix = Image.open("model_evaluation/support_vector_classifier/svc_confusion_matrix.png")
				st.image(conf_matrix, use_container_width=False)
			
            ### Display ROC curve
			with st.expander("Receiver Operating Characteristic (ROC) Curve"):
				roc_curve = Image.open("model_evaluation/support_vector_classifier/svc_roc_curve.png")
				st.image(roc_curve, use_container_width=False)
				st.success("**Higher the AUC, better the ability of the model to separate the two classes, regardless of the threshold.**")
