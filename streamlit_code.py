import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import DictVectorizer
import numpy as np
vec = DictVectorizer(sparse=False, dtype=np.int64)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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
high = unique_df[unique_df['category'] == 'High-importance']
low = unique_df[unique_df['category'] == 'Low-importance']
mid = unique_df[unique_df['category'] == 'Mid-importance']

all_df = load_data("all_health_articles.csv")
all_df['date'] = pd.to_datetime(all_df['date'])
df_2023 = all_df[all_df['date'].dt.year == 2023]
df_2024 = all_df[all_df['date'].dt.year == 2024]


# creating 6 tabs for each section 
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Main Page & Findings", "Data Summary", "New Features", "Feature Engineering", "Hypothesis Testing", "Summary and Ethical Considerations"])

with tab1: 
    st.header("CS234 Final Project: Classifying & Comparing Health Articles by Level of Importance")
    st.markdown("""
                **Background**: There are two motivations for this project. 1) With the rise of anti-science/anti-establishment rhetoric deployed against health & wellness in the U.S., as well as the recent proliferation of discourse around health improvement on social media, I am curious to explore how Wikipedia articles on health are viewed, and if they reflect or uncover similar or different patterns to how health is discussed on other platforms. 2) For many WikiProjects, I noticed that there are several assessments for each article, including but not limited to quality and level of importance. While we have briefly explored the quality metric for Project 1, I was interested in finding a way to quantify or measure the level of importance of an article because there are many unlabeled articles (i.e., the level of importance was not assigned to the article) and subjective labeling is a time-consuming process that is difficult to replicate. Therefore, I combine these ideas to investigate levels of importance for Wikipedia articles on health.\n
                **Research Question**: How have total pageviews under the topic of “Health and Fitness” changed over time? How, if at all, do the pageviews of articles across different levels of importance differ between each other? \n
                **Expectations**: The goal of this research project is to 1) classify articles by the level of importance and 2) test if there is a difference between pageviews across time & across level of importance. \n   
                **Ethical considerations & Limitations**: The scope of the dataset is limited geographically to the United States and linguistically to the English language. The obscure nature of assessing the level of importance for articles in each Wikiproject makes it especially difficult to tackle this task of assessment from an equitable standpoint. For instance, Healthcare in the United States is labeled as a high-importance article, but not Healthcare in any other region or country. It’s pivotal to consider what a high or low level of importance assignation can alter engagement with these subjects too. Even though there are more supposedly objective metrics in determining importance, such as the total number of revisions, it’s still important to keep in mind the skewed interest towards larger countries or establishments. \n
    """)

    st.subheader("Presentation of Findings")

with tab2:
    st.header("Data Summary")
    st.markdown("""
                **Source**: I used the project page containing all [Health and Fitness Articles]("https://en.wikipedia.org/wiki/Category:Health_and_fitness_articles_by_importance") (labeled by level of importance). There are 6 subcategories: high, medium, low, top importance & NA/unknown importance, with a total of 6904 articles. I assembled a list of QIDs for these articles using the mwclient module, and ended up with a list of 3177 QIDs. \n
                **Scope**: The fully assembled dataset contains all pageviews associated with Health and Fitness articles in the English language and in the United States from 2023 to 2024. I talked to the DuckDB server for both the 2023 and 2024 datasets and filtered them with the unique QID list to only preserve relevant articles and their associated daily pageviews. 
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Number of Rows", f"{len(all_df)}")
        st.metric("Total Unique Articles", f"{len(unique_df)}")
    with col2:
        st.metric("Total Pageviews", f"{unique_df['total_pageviews'].sum():,}")
        st.metric("2023 Pageviews", f"{df_2023['pageviews'].sum():,}")
        st.metric("2024 Pageviews", f"{df_2024['pageviews'].sum():,}")

    st.subheader("Process of Assembling Data")
    st.markdown("""
                1. Get all talk pages for each category.
                2. Navigate to the actual page with substance and save page titles.
                3. Extract relevant identifier properties: QID.
                4. Repeat for all categories.
                5. Save all data into a json file.

    """)
    st.subheader("Data Overview")
    st.write("Click below to see which articles are labeled 'high importance' or 'low importance'!")
    with st.expander("Show high-level importance data!"):
        high = high.drop('category', axis=1)
        high = high.sort_values(by='total_pageviews', ascending=False)
        st.dataframe(high.head())
    
    with st.expander("Show low-level importance data!"):
        low = low.drop('category', axis=1)
        low = low.sort_values(by='total_pageviews', ascending=False)
        st.dataframe(low.head())
    
    st.write("Articles by Category")


    
with tab3: 
    st.header("New Features & Feature Engineering")
    # explain input data for level of importance 
    st.markdown("To do feature engineering later, I needed to acquire many more new features beyond the given article title, QID, and category. Determining the exact features I needed was a tedious process: I consulted multiple pages on [how Wikipedia editors assess the level of importance of an article](https://en.wikipedia.org/wiki/Wikipedia:Assessing_articles), and with Professor Eni’s help, I decided on these additional features to include: ")
    st.markdown("""
                1. page size
                2. number of incoming & outgoing links
                3. age of the article
                4. total number of revisions
                5. total number of unique editors
                6. unique number of days viewed. 
                """)
    st.markdown("All of these metrics are relevant to understanding the supposed status of an article within the given context of a Wikiproject, and answers the core question: **'How important is it to Wikipedia's coverage of this project's subject area that there should be an article for this topic?'**")

    st.subheader("Exploration of Features")
    
    size_fig = px.box(
        unique_df.sort_values(by='page_size'),
        x='category',
        y='page_size',
        title='Distribution of Page Size by Article Importance',
        labels={'category': 'Importance Category', 'page_size': 'Page Size'}
    )
    st.plotly_chart(size_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        sorted_revisions = unique_df.sort_values(by='total_revisions')
        revisions_fig = px.box(sorted_revisions, x='category', y='total_revisions', title = "Number of Revisions")
        st.plotly_chart(revisions_fig, use_container_width=True)
    with col2:
        editors_fig = px.box(sorted_revisions, x='category', y='num_editors', title = "Number of Unique Editors")
        st.plotly_chart(editors_fig, use_container_width=True)
    
    mean_age_df = unique_df.groupby('category')['article_age'].mean()
    age_fig = px.bar(mean_age_df,y='article_age', title='Mean Article Age', labels={"article_age": "Age of Articles (in Days)"})
    st.plotly_chart(age_fig, use_container_width=True)

    st.header("Feature Engineering")
    st.markdown("Instead of text classification, I chose to do feature engineering because my purpose was to build an accurate, applicable classifier that could best label unlabeled articles; this would be a helpful contribution to the categorization and structuring of Wikiprojects. ")
    st.divider()
    st.markdown("1. Before beginning vectorization, I removed the unknown-importance articles since they might add too much confusing noise to the training.")
    classifier_df = unique_df[unique_df['category'] != 'Unknown-importance']
    classifier_df = classifier_df.fillna(0)
    st.metric("Total number of Labeled Unique Articles on Health & Fitness:", f"{len(classifier_df):,}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of High Importance Articles:", f"{len(high):,}")
    with col2:
        st.metric("Number of Low Importance Articles:", f"{len(low):,}")
    with col3:
        st.metric("Number of Mid Importance Articles:", f"{len(mid):,}")

    st.divider()
    st.markdown("2. I vectorized the features and labels (both of which are matrices), then split the dataset using train_test_split (test size = 0.1, because we have a small dataset of 750 rows and should try to train with a considerable portion of this set).")
    onlyFeatures = classifier_df[["total_pageviews", "page_size", "incoming_links", "outgoing_links", "num_editors", "article_age", "total_revisions", "total_translations", "unique_days_viewed"]]
    dfDict = onlyFeatures.to_dict('records')
    X = vec.fit_transform(dfDict)
    y = classifier_df['category']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                      test_size=0.1,  
                                                      random_state=42) 

    split_code = """
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y,
                                                            test_size=0.1,  
                                                            random_state=42) 

    """
    st.code(split_code, language="Python")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Size of Training Dataset", f"{675}")
    with col2:
        st.metric("Size of Testing Dataset", f"{75}")
    
    st.divider()
    st.markdown("3. Finally, since I had 3 classes for prediction, I built a multinominal logistic regression classifier. ")
    st.markdown("NOTE: There was a suggestion from sklearn that I could preprocess the data and scale it up to improve accuracy, so I followed that and used a StandardScaler. ")
    scaling_code = """
    from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Fit on training data and transform both train and test
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        """
    st.code(scaling_code, language="python")

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # Fit on training data and transform both train and test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    st.subheader("Confusion Matrix")
    labels = sorted(unique_df['category'].unique())
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Multinomial Logistic Regression)')
    st.pyplot(fig)


with tab4: 
    st.header("Feature Engineering")

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