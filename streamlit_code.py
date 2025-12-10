import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Crystal's Final Project (in Process): Health Article Classification by Importance",
    layout="wide"
)

st.title("Health Article Classification by Importance")
st.markdown("""
The dataset contains all pageviews associated with health articles (n=3177) that are either labeled as high, medium, or low level importance.
The scope of the dataset is limited to the U.S. in the English language from 2023 to 2024.  

The goal of this research project is to 1) classify articles by the level of importance and 2) test if there is a difference between pageviews across time & across level of importance.       
""")

st.header("1. Assembling Full Dataset")
st.markdown("There are 6 subcategories: high, medium, low, top importance & NA/unkown importance.")
st.markdown("Each category is associated with a talk page. Using the talk page, I then accessed the page properties via mwClient API calls. I finally talked to the duckdb server and filtered the full dataset using this unique QID list.")
st.markdown("In the future, I will try to reassemble this dataset to also retrieve other properties like page size and number of unique revisions, as these are helpful features that can be used to build a classifier.")

# UNIQUE HEALTH ARTICLES Section
@st.cache_data
def load_data_1():
    try:
        # Load the uploaded file
        df1 = pd.read_csv("unique_health_articles.csv")
        
        # Remove "Unnamed: 0" column if it exists (common artifact in CSVs)
        if "Unnamed: 0" in df1.columns:
            df1 = df1.drop(columns=["Unnamed: 0"])
            
        # Ensure correct data types
        df1['total_pageviews'] = pd.to_numeric(df1['total_pageviews'], errors='coerce').fillna(0)
        df1['description'] = df1['description'].fillna("")
        
        return df1
    except FileNotFoundError:
        st.error("File 'unique_health_articles.csv' not found. Please upload it.")
        return pd.DataFrame()

df1 = load_data_1()
st.write("Unique health articles - raw data overview:", df1.head())
st.metric("Total Articles", f"{len(df1)}")
st.metric("Total Pageviews", f"{df1['total_pageviews'].sum():,}")
st.metric("Average Pageviews / Article", f"{df1['total_pageviews'].mean():,.0f}")


st.subheader("Pageview Across Category Analysis")
cat_counts = df1['category'].value_counts().rename_axis('Category').reset_index(name='Count')
    
fig_bar = px.bar(
    cat_counts, 
    x='Category', 
    y='Count', 
    color='Category',
    text='Count',
    title="Number of Articles per Category",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig_bar.update_traces(textposition='outside')
st.plotly_chart(fig_bar, use_container_width=True)


# ALL HEALTH ARTICLES by date section
def load_data_2():
    try:
        # Load the uploaded file
        df2 = pd.read_csv("all_health_articles.csv")
        
        return df2
    except FileNotFoundError:
        st.error("File not found. Please upload it.")
        return pd.DataFrame()
df2 = load_data_2()
df2['date'] = pd.to_datetime(df2['date'])


st.subheader("Time Series Analysis: Pageviews by Category")
all_cats = sorted(df2['category'].unique())
selected_cats = st.multiselect(
    "Select Categories to Compare",
    options=all_cats,
    default=all_cats # Select all by default
)
agg_type = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=1, key="granularity_cat")

# We sum pageviews for ALL articles within the same category for the given time period
if agg_type == "Daily":
    # Group by Date AND Category
    plot_df = df2.groupby(['date', 'category'])['pageviews'].sum().reset_index()
    
elif agg_type == "Weekly":
    plot_df = (
        df2
        .set_index('date')
        .groupby('category')
        .resample('W')['pageviews']
        .sum()
        .reset_index()
    )
    
elif agg_type == "Monthly":
    plot_df = (
        df2
        .set_index('date')
        .groupby('category')
        .resample('M')['pageviews']
        .sum()
        .reset_index()
    )

# --- 4. Visualization ---

if not plot_df.empty:
    st.subheader(f"Total Pageviews by Category ({agg_type})")
    
    fig = px.line(
        plot_df, 
        x='date', 
        y='pageviews', 
        color='category', # Different line for each category
        markers=True,
        title="Aggregate Pageviews per Category",
        template="plotly_white"
    )
    
    fig.update_traces(hovertemplate='%{y:,.0f} views')
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary Metrics
    st.divider()
    cols = st.columns(len(selected_cats))
    for i, cat in enumerate(selected_cats):
        total_views = plot_df[plot_df['category'] == cat]['pageviews'].sum()
        cols[i].metric(label=cat, value=f"{total_views:,.0f}")

else:
    st.info("No data available for the selected filters.")



st.subheader("Time Series Analysis: Aggregated Pageviews Over Time")
agg_type = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=1, key="granularity_agg")

# Step 2: Aggregate
# We sum pageviews for ALL articles within the same category for the given time period
if agg_type == "Daily":
    # Group by Date AND Category
    plot_df = df2.groupby('date')['pageviews'].sum().reset_index()
    
elif agg_type == "Weekly":
    plot_df = (
        df2
        .set_index('date')
        .resample('W')['pageviews']
        .sum()
        .reset_index()
    )
    
elif agg_type == "Monthly":
    plot_df = (
        df2
        .set_index('date')
        .resample('M')['pageviews']
        .sum()
        .reset_index()
    )

# --- 4. Visualization ---

if not plot_df.empty:
    st.subheader("Pageviews Across Time")
    
    fig = px.line(
        plot_df, 
        x='date', 
        y='pageviews', 
        markers=True,
        title="Aggregate Pageviews",
        template="plotly_white"
    )
    
    fig.update_traces(hovertemplate='%{y:,.0f} views')
    st.plotly_chart(fig, use_container_width=True)
