import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
 

def load_viz_library():
	global sns, px
	import seaborn as sns
	import plotly.express as px

@st.cache_data
def load_data(data):
	df = pd.read_csv(data)
	return df

def run_eda_app():
	load_viz_library()
	st.subheader("Exploratory Data Analysis Section")

    # Load the dataset
	df = load_data("data/diabetes_data_upload.csv")		# relative path
	df_clean = load_data("data/diabetes_data_clean.csv")
	freq_df = load_data("data/frequency_distribution_age.csv")

	df_new = df.copy()
	df_new.columns = df_new.columns.str.lower().str.replace(' ', '_')

	submenu = st.sidebar.selectbox("Sub Menu",["Descriptive Analysis","Plots"])
	
    ######### Go to Descriptive Analysis subsection ########################
	if submenu == "Descriptive Analysis":
		st.subheader("Descriptive Analysis")

        # Show original dataset
		with st.expander("Original Dataset"):
			st.dataframe(df)
		
        # Show data types of original dataset
		with st.expander("Data Types of Original Dataset"):
			dtypes_df = df.dtypes.reset_index()
			dtypes_df.columns = ['Feature', 'Data Type']
			dtypes_df.set_index("Feature", inplace=True)
			st.write(dtypes_df)
			
        # Show the shape of original dataset
		with st.expander("Dataset Shape"):
			st.write(f"Shape of the dataset: **{df_clean.shape}**")
			st.write(f"Number of observations (or rows): **{df_clean.shape[0]}**")
			st.write(f"Number of features (or columns): **{df_clean.shape[1]}**")

        # Show any missing values from original dataset
		with st.expander("Missing Value Counts from the Original Dataset"):
			missing_value_df = pd.DataFrame(df.isnull().sum())
			missing_value_df = missing_value_df.reset_index()
			missing_value_df.columns = ['Feature', 'Missing Value']
			missing_value_df.set_index('Feature', inplace=True)
			st.dataframe(missing_value_df, width=300)
			st.write("**As it can be seen, there are no missing values in any column of the dataset.**")

        # Show the cleaned dataset
		with st.expander("Cleaned Dataset"):
			st.dataframe(df_clean)
			st.write("**Note:** Here, the consistent column names are used.\n\n" \
			"The values of all the columns also have been encoded, except the 'age' column.\n\n" \
			"**Encodings Applied:**\n\n" \
			"{'Female': 0, 'Male': 1}\n\n" \
			"{'No': 0, 'Yes': 1}\n\n" \
			"{'Negative': 0, 'Positive': 1}")

        # Show data types of cleaned dataset
		with st.expander("Data Types of Cleaned Dataset"):
			dtypes_df = df_clean.dtypes.reset_index()
			dtypes_df.columns = ['Feature', 'Data Type']
			dtypes_df.set_index("Feature", inplace=True)
			st.write(dtypes_df)

        # Show descriptive summary
		with st.expander("Descriptive Summary"):
			st.dataframe(df_clean.describe())
		
        # Show the gender distribution dataframe
		with st.expander("Gender Distribution"):
				st.dataframe(df_new['gender'].value_counts())

        # Show the class distribution dataframe
		with st.expander("Class Distribution"):
			st.dataframe(df_new['class'].value_counts())

    ################ Go to Plots subsection ########################
	else:
		st.subheader("Plots")

        # Show the distribution plot of gender
		with st.expander("Distribution Plot of Gender"):
			gen_df = df_new['gender'].value_counts().to_frame()
			gen_df = gen_df.reset_index()
			gen_df.columns = ['Gender Type','Counts']
			p01 = px.pie(gen_df,names='Gender Type',values='Counts')
			st.plotly_chart(p01,use_container_width=True)

        # Show the distribution plot of class
		with st.expander("Distribution Plot of Class"):
			fig = plt.figure()
			sns.countplot(df['class'])
			st.pyplot(fig)
		
        # Show the distribution plot of polyuria
		with st.expander("Distribution Plot of Polyuria"):
			fig = plt.figure()
			sns.countplot(x=df['Polyuria'],hue=df['class'], data=df, palette='Set1')
			st.pyplot(fig)
		
        # Show the distribution plot of polydipsia
		with st.expander("Distribution Plot of Polydipsia"):
			fig = plt.figure()
			sns.countplot(x=df['Polydipsia'],hue=df['class'], data=df, palette='Set2')
			st.pyplot(fig)

        # Show the frequency distribution plot of age
		with st.expander("Frequency Distribution Plot of Age"):
			plot = px.bar(freq_df,x='age',y='count')
			st.plotly_chart(plot)

        # Show the detected outliers
		with st.expander("Outlier Detection Plot"):
			p3 = px.box(df_new,x='age',color='gender')
			st.plotly_chart(p3)

        # Show the correlation plot
		with st.expander("Correlation Plot"):
			corr_matrix = df_clean.corr()
			fig = plt.figure(figsize=(20,10))
			sns.heatmap(corr_matrix,annot=True)
			st.pyplot(fig)