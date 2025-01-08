import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load and preprocess the dataset
@st.cache_data
def load_data():
    # Replace with your dataset path
    data = pd.read_csv("C:/Users/Admin/Downloads/user_personalized_features.csv")  

    # Encode categorical features
    label_encoder = LabelEncoder()
    for col in ['Gender', 'Location', 'Interests', 'Product_Category_Preference']:
        data[col] = label_encoder.fit_transform(data[col])

    # Normalize numeric columns
    numeric_cols = ['Age', 'Income', 'Last_Login_Days_Ago', 'Purchase_Frequency', 
                    'Average_Order_Value', 'Total_Spending', 'Time_Spent_on_Site_Minutes', 'Pages_Viewed']
    scaler = MinMaxScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# Recommendation function
def recommend_content_based(user_id, data, top_n=5):
    features = ['Interests', 'Product_Category_Preference', 'Age', 'Gender', 'Income', 
                'Time_Spent_on_Site_Minutes', 'Pages_Viewed']
    user_profiles = data[features]
    similarity_matrix = cosine_similarity(user_profiles)

    # Find recommendations
    user_index = data[data['User_ID'] == user_id].index[0]
    similarity_scores = list(enumerate(similarity_matrix[user_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top similar users (excluding self)
    top_users = [data.iloc[x[0]] for x in similarity_scores[1:top_n+1]]
    return pd.DataFrame(top_users)[['User_ID', 'Product_Category_Preference']]

# Streamlit App
def main():
    st.title("E-commerce Product Recommendation System")
    st.write("This app recommends products based on user preferences.")

    # Load data
    data = load_data()

    # User input for recommendation
    user_id = st.selectbox("Select a User ID:", data['User_ID'].unique())

    if st.button("Get Recommendations"):
        recommendations = recommend_content_based(user_id, data, top_n=5)
        st.subheader("Recommended Products:")
        st.table(recommendations)

if __name__ == "__main__":
    main()
