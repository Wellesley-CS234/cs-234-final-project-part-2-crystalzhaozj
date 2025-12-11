import streamlit as st
import pandas as pd
import plotly.express as px
import scipy
from scipy import stats

st.set_page_config(
    page_title="Crystal's Final Project (in Process): Health Article Classification by Importance",
    layout="wide"
)

@st.cache_data
def load_data(csvfile):
    try:
        df = pd.read_csv(csvfile)
        return df
    except FileNotFoundError:
        st.error("File not found. Please upload it")
        return pd.DataFrame()

# Loading both data files 
unique_df = load_data("unique_health_articles.csv")
unique_df = unique_df.loc[:, ~unique_df.columns.str.contains('^Unnamed')]
unique_df['total_pageviews'] = pd.to_numeric(unique_df['total_pageviews'], errors='coerce').fillna(0)
unique_df['description'] = unique_df['description'].fillna("")

all_df = load_data("all_health_articles.csv")
all_df['date'] = pd.to_datetime(all_df['date'])

# creating 6 tabs for each section 
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Introduction", "Data Summary", "New Features", "Text Classification", "Hypothesis Testing", "Summary and Ethical Considerations"])

with tab1: 
    st.header("CS234 Final Project: Classifying & Comparing Health Articles by Level of Importance")
    st.markdown("""
                **Background**: 
                **Research Question**: 
                **Expectations**: The goal of this research project is to 1) classify articles by the level of importance and 2) test if there is a difference between pageviews across time & across level of importance.       
    """)

with tab2:
    st.header("Data Summary")
    st.markdown("""
                **Source**: I used the project page containing all [Health and Fitness Articles]("https://en.wikipedia.org/wiki/Category:Health_and_fitness_articles_by_importance") (labeled by level of importance). There are 6 subcategories: high, medium, low, top importance & NA/unkown importance. There should be a total of 6904 articles.
                **Scope**: The fully assembled dataset contains all pageviews associated with Health and Fitness articles in the English language and in the United States from 2023 to 2024. 
    """)
    st.subheader("Process of Assembling Data")
    st.markdown("""

    """)
    st.write("Here is a snipppet of the dataset.")
    unique_df.head()
    st.metric("Total Articles", f"{len(unique_df)}")
    st.metric("Total Pageviews", f"{unique_df['total_pageviews'].sum():,}")
    st.metric("Average Pageviews / Article", f"{unique_df['total_pageviews'].mean():,.0f}")
    
with tab3: 
    st.header("New Features")
    # explain input data for level of importance 

with tab4: 
    st.header("Text Classification")

with tab5:
    st.header("Hypothesis Testing")
    st.subheader("Hypothesis: There is no significant difference between pageviews for health articles across categories of high, medium, low, and unknown levels of importance.")

    # for a small sample: use p-value testing
    # for everything: use visualizations

with tab6: 
    st.header("Summary and Ethical Considerations")
st.header("1. Assembling Full Dataset")
st.markdown("There are 6 subcategories: high, medium, low, top importance & NA/unkown importance.")
st.markdown("Each category is associated with a talk page. Using the talk page, I then accessed the page properties via mwClient API calls. I finally talked to the duckdb server and filtered the full dataset using this unique QID list.")
st.markdown("In the future, I will try to reassemble this dataset to also retrieve other properties like page size and number of unique revisions, as these are helpful features that can be used to build a classifier.")


"""
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
"""