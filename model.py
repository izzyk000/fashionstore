import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# Load the trained Keras model
model = load_model('churn_predictor_model.keras')

def preprocess_new_data(data):
    """ Preprocess the new data similar to the training data """
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data['Geographical Location'] = label_encoder.fit_transform(data['Geographical Location'])

    scaler = StandardScaler()
    numerical_features = ['Time Since Last Purchase', 'Total Number of Orders', 'Purchase Frequency',
                          'Number of Website Visits', 'Number of Items Viewed', 'Age', 'Total Spend']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

def predict(data):
    """ Predict churn using the preprocessed data """
    predictions = model.predict(data)
    return np.round(predictions).astype(int)

def plot_feature_churn_correlation(data):
    """ Create and plot correlation heatmap between features and churn likelihood using Plotly """
    # Ensure only numeric data is considered and exclude 'Cluster'
    numeric_data = data.select_dtypes(include=[np.number])
    if 'Cluster' in numeric_data.columns:
        numeric_data = numeric_data.drop(columns=['Cluster'])
    # Calculate the correlation matrix for numeric data
    corr = numeric_data.corr()
    # Plot only the correlation values related to 'Churn Prediction'
    fig = px.imshow(corr[['Churn Prediction']], text_auto=True, aspect="auto",
                    color_continuous_scale='Blues', title="Feature Correlation with Churn Likelihood")
    return fig


def perform_clustering(data):
    """ Perform clustering and assign consistent cluster labels based on 'Total Spend' and 'Purchase Frequency' """
    scaler = StandardScaler()
    clustering_features = ['Total Spend', 'Purchase Frequency']
    scaled_features = scaler.fit_transform(data[clustering_features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)

    # Map clusters to fixed labels based on predefined conditions
    cluster_centers = kmeans.cluster_centers_
    data['Cluster'] = cluster_labels

    # Re-label clusters based on conditions
    data['Cluster'] = data.apply(
        lambda row: 'Low Spend' if row['Total Spend'] < 0 and row['Purchase Frequency'] < 0 else (
                    'High Spend, Low Frequency' if row['Total Spend'] > 0 and row['Purchase Frequency'] < 0 else 
                    'High Spend, High Frequency'), axis=1)

    return data



def plot_customer_segmentation(data):
    """ Plot Customer Segmentation with Churn Rate """
    symbol_map = {'Low Spend': 'circle', 'High Spend, Low Frequency': 'square', 'High Spend, High Frequency': 'diamond'}
    color_map = {'No Churn': 'blue', 'Churn': 'red'}  # Custom color map

    fig = px.scatter(data, x='Total Spend', y='Purchase Frequency', color='Churn Label',
                     symbol='Cluster', labels={'Churn Label': 'Churn Prediction', 'Cluster': 'Customer Segment'},
                     title='Customer Segmentation with Churn Rate', symbol_map=symbol_map, 
                     color_discrete_map=color_map)
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=2, color='DarkSlateGrey')))
    return fig



# Streamlit interface
st.sidebar.title('About Customer Churn Predictor')
st.sidebar.write("""
This predictive tool is designed exclusively for StyleVogue, a leading online clothing retailer in the USA. By analyzing extensive customer data, our model assesses factors like purchase frequency, website engagement, and spending behavior to predict the likelihood of customer churn. This helps StyleVogue identify at-risk customers and implement targeted retention strategies effectively.

Key Features:

- **Churn Likelihood**: Predicts the probability of a customer churning based on their interaction history and spending behavior.
- **Customer Segmentation**: Groups customers into segments based on their spend and purchase frequency to better understand their behavior and target retention strategies effectively.

Developed by Data Doodles, this tool empowers businesses to make data-driven decisions to enhance customer retention and drive sales growth.
""")

st.title('Customer Churn Prediction')
st.write("""
Welcome to the Customer Churn Prediction tool! This sophisticated AI model evaluates customer profiles to determine the likelihood of them discontinuing service at your clothing store.

Factors Considered for Prediction:

- Demographic and Background: Gender, Geographical Location, Age
- Shopping Behavior: Time Since Last Purchase, Total Number of Orders, Purchase Frequency
- Engagement: Number of Website Visits, Number of Items Viewed
- Financial: Total Spend

The model evaluates customers by considering a comprehensive set of factors, including demographic information, shopping behavior, engagement, and financial data. The predictive analysis focuses on:

- Churn Likelihood: Predicts the probability of a customer churning based on their interaction history and spending behavior.
- Customer Segmentation: Groups customers into segments based on their spend and purchase frequency to better understand their behavior and target retention strategies effectively.

How to Use This Tool:

1. **Upload Customer Data**: Upload an Excel file with customer details.
2. **Predict Churn**: Click the 'Predict' button to process the data and generate churn predictions.
3. **Review the Results**: Examine the prediction outcome and customer segmentation to make informed decisions for customer retention.

Model Accuracy: 96%
""")

uploaded_file = st.file_uploader("Choose a file", type=['xlsx'])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.write('Uploaded Data:')
    st.write(data)
    preprocessed_data = preprocess_new_data(data)

    if st.button('Predict Churn for Uploaded Data'):
        predictions = predict(preprocessed_data)
        data['Churn Prediction'] = predictions  # Keep numeric predictions for analysis
        data['Churn Label'] = data['Churn Prediction'].apply(lambda x: 'Churn' if x == 1 else 'No Churn')
        st.write('Prediction Results:')
        st.write(data[['Churn Label']])
        data = perform_clustering(data)  # Make sure clustering is performed and modifies the DataFrame

        churn_distribution = data['Churn Prediction'].value_counts(normalize=True) * 100
        fig_pie = px.pie(churn_distribution, names=churn_distribution.index, values=churn_distribution.values,
                         title='Churn vs No Churn Percentage', hole=0.4)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie)

        with st.expander("Understand the Churn Prediction Results"):
            st.markdown("""
            **Detailed Analysis of Churn Prediction Results:**

            - **Overview**: The pie chart above represents the percentage distribution between customers predicted to churn and those predicted to not churn. This visualization helps in quickly identifying the proportion of at-risk customers.

            - **Churn Predictions**: Indicates customers likely to discontinue services. A higher percentage here suggests that more customers are at risk, signaling a need for proactive retention strategies.

            - **No Churn Predictions**: Represents customers likely to continue with the services. A higher proportion of 'No Churn' suggests stable customer loyalty and lower immediate risk of losing customers.

            - **Implications for Business Strategies**:
                - **Targeted Interventions**: For segments showing higher churn predictions, targeted interventions such as personalized offers, improved service delivery, or engagement strategies could be employed to improve retention.
                - **Customer Satisfaction and Feedback**: Engaging with customers, especially those predicted to churn, to gather feedback and understand their concerns can provide insights into potential areas of improvement.
                - **Resource Allocation**: Understanding the churn predictions helps in optimizing resource allocation towards customer retention strategies and operational adjustments.

            **Action Points**:
            - **Engage with At-Risk Customers**: Develop programs aimed at improving customer experience for those predicted to churn.
            - **Analyze Customer Journey**: Look for common patterns among customers predicted to churn and address any service gaps or friction points.
            - **Monitor and Adjust**: Regularly review the model's predictions with actual churn rates and refine strategies and the model itself to improve accuracy and effectiveness.
            """)

        st.plotly_chart(plot_feature_churn_correlation(data))
        with st.expander("Understand the Feature Correlation with Churn Likelihood"):
            st.markdown("""
            **Detailed Analysis of Feature Correlation with Churn Likelihood:**

            - **Overview**: The heatmap displayed above provides insights into how each feature correlates with the likelihood of churn. This visualization highlights the degree and direction of association between each feature and churn.

            - **Understanding Correlation Values**:
                - **Positive Correlation**: When a feature has a positive correlation with churn, it means that as the value of the feature increases, the likelihood of churn also increases. For example, 'Purchase Frequency' shows a positive correlation, suggesting that more frequent purchases might be associated with a higher likelihood of churn, potentially indicating customer dissatisfaction or competitive pricing issues.
                - **Negative Correlation**: Conversely, a negative correlation means that as the value of the feature increases, the likelihood of churn decreases. This can be observed in 'Total Spend', where higher spending is associated with a lower chance of churn, possibly reflecting higher customer engagement or satisfaction with the products.

            - **Implications for Business Strategies**:
                - **Data-Driven Decisions**: Utilize the insights from the correlation analysis to make informed decisions about which aspects of customer relationships need attention.
                - **Enhance Predictive Modeling**: Adjust the feature selection for machine learning models based on their correlation with churn to improve predictive accuracy.
                - **Strategic Resource Allocation**: Allocate resources more efficiently by focusing on influencing factors that are strongly correlated with churn.

            - **Action Points**:
                - **Focus on Key Drivers**: Enhance or modify features that are strongly correlated with churn to better manage customer retention.
                - **Reassess Feature Importance**: Regularly update the analysis to reflect changes in trends and customer behavior, ensuring that the model remains effective over time.
                - **Customize Customer Interactions**: Use insights from the analysis to customize interactions with customers, addressing key factors that influence their decision to stay or leave.

            - **Analytical Insights**:
                - **Interpretation of Positive Values**: Positive correlations highlight risk areas where increasing values typically contribute to customer churn. For example, increasing 'Purchase Frequency' might require analysis to determine if it reflects negative experiences or other issues.
                - **Interpretation of Negative Values**: Negative correlations identify protective factors where increasing values are beneficial and potentially reduce churn. For instance, increasing 'Total Spend' may indicate a loyal customer base that finds value in the products offered.
            """)

        st.write('Prediction and Segmentation Results:')

        # Visualization of Customer Segments with Churn Rate
        st.plotly_chart(plot_customer_segmentation(data))

        # Optional: Detailed Analysis Expander
        with st.expander("Understand the Customer Segmentation with Churn Rate"):
            st.markdown("""
            **Detailed Analysis of Customer Segmentation with Churn Rate:**

            - **Overview**: This scatter plot visually segregates customers into distinct segments based on their 'Total Spend' and 'Purchase Frequency'. Each shape in the chart represents a different segment, while the colors indicate whether customers in that segment are predicted to churn or not.

            - **Understanding the Shapes**:
                - **Circles**: Indicate customers in Segment 0, which might consist of customers with lower spending and irregular purchase patterns, potentially at a higher risk of churning.
                - **Squares**: Represent customers in Segment 1. This segment typically includes customers with moderate spending and purchase frequency. The size and color of the squares may vary to reflect the spending amount and churn likelihood respectively.
                - **Diamonds**: Represent customers in Segment 2, often characterized by high spending and frequent purchases. These are likely your most engaged customers.

            - **Color Coding**:
                - **Red**: Indicates customers who are likely to churn. Customers in this category might require immediate attention through retention strategies.
                - **Blue**: Represents customers predicted not to churn, suggesting stable loyalty and potentially higher customer satisfaction within this group.

            - **Implications for Business Strategies**:
                - **Targeted Interventions**: Understanding which segment a customer belongs to can help tailor marketing strategies. For example, customers in red diamonds might benefit from loyalty programs to reduce churn risk.
                - **Resource Allocation**: Allocating resources effectively according to the segmentation can optimize marketing efforts and improve customer retention rates.
                - **Customer Engagement**: Enhancing engagement with at-risk segments (red circles and squares) through personalized communication and offers might improve their retention rates.

            - **Action Points**:
                - **Engage with At-Risk Customers**: Develop specific programs aimed at improving customer experience for those in high-risk segments (e.g., red squares).
                - **Analyze Segment Needs**: Conduct surveys or focus groups within each segment to better understand their needs and improve service offerings accordingly.
                - **Monitor and Adjust**: Regularly review the segmentation and churn predictions to refine strategies and adapt to changing customer behaviors.
            """)
