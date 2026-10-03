
# ============================================================
# STEP 1: INSTALL LIBRARIES AND LOAD DATASET
# ============================================================


import pandas as pd
import numpy as np
import plotly.express as px
import gradio as gr
import re
import os

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------

FILE_PATH = "/content/amazon_prime_movies_tv_2025_EDA_ready.csv"

df = pd.read_csv("amazon_prime_movies_tv_2025_EDA")

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns:")
for column in df.columns:
    print("-", column)

display(df.head())


# ============================================================
# STEP 2: DATA CLEANING AND PREPARATION
# ============================================================

data = df.copy()

# ------------------------------------------------------------
# Remove duplicate records
# ------------------------------------------------------------

data = data.drop_duplicates().reset_index(drop=True)

# ------------------------------------------------------------
# Clean column names
# ------------------------------------------------------------

data.columns = data.columns.str.strip()

# ------------------------------------------------------------
# Clean important text columns
# ------------------------------------------------------------

text_columns = [
    "Title",
    "Genre(s)",
    "Type",
    "Language",
    "Country of origin",
    "Age Rating",
    "Content Category"
]

for col in text_columns:
    data[col] = (
        data[col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

# ------------------------------------------------------------
# Convert numerical columns
# ------------------------------------------------------------

numeric_columns = [
    "Release Year",
    "IMDb Score",
    "Number of IMDb Votes",
    "Number of episodes",
    "Duration Value"
]

for col in numeric_columns:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

# ------------------------------------------------------------
# Clean Type
# ------------------------------------------------------------

data["Type"] = (
    data["Type"]
    .str.strip()
    .str.title()
)

# ------------------------------------------------------------
# Clean Content Category
# ------------------------------------------------------------

data["Content Category"] = (
    data["Content Category"]
    .str.strip()
    .str.title()
)

# ------------------------------------------------------------
# Convert availability to boolean
# ------------------------------------------------------------

data["Is Available"] = (
    data["Is Available"]
    .astype(str)
    .str.lower()
    .map({
        "true": True,
        "false": False
    })
    .fillna(False)
)

# ------------------------------------------------------------
# Create a list of individual genres
# ------------------------------------------------------------

def split_genres(value):

    if pd.isna(value):
        return ["Unknown"]

    value = str(value)

    # Handles:
    # Drama, Comedy
    # Drama & Comedy
    # Drama / Comedy
    # Drama | Comedy

    genres = re.split(
        r",|&|/|\|",
        value
    )

    genres = [
        g.strip()
        for g in genres
        if g.strip()
    ]

    return genres if genres else ["Unknown"]


data["Genre List"] = data["Genre(s)"].apply(
    split_genres
)

# ------------------------------------------------------------
# Create genre-expanded dataframe
# ------------------------------------------------------------

genre_data = data.explode(
    "Genre List"
).copy()

genre_data["Genre"] = (
    genre_data["Genre List"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

# ------------------------------------------------------------
# Display information
# ------------------------------------------------------------

print("Cleaning completed!")
print("Final rows:", len(data))

print("\nMovie types:")
print(data["Type"].value_counts())

print("\nLanguages:")
print(data["Language"].value_counts().head(10))

print("\nGenres:")
print(genre_data["Genre"].value_counts().head(10))

display(data.head())
# ============================================================
# STEP 3
# RECOMMENDATION ENGINE + AGE RATING + AMAZON SEARCH LINK
# ============================================================

import urllib.parse


# ============================================================
# DROPDOWN VALUES
# ============================================================

GENRES = sorted(
    genre_data["Genre"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

ORIGINS = sorted(
    data["Country of origin"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

LANGUAGES = sorted(
    data["Language"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

CATEGORIES = sorted(
    data["Content Category"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

AGE_RATINGS = sorted(
    data["Age Rating"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# ============================================================
# AGE RATING INFORMATION
# ============================================================

AGE_RATING_INFO = {

    "U": "U — Suitable for all ages",
    "UA": "UA — Suitable for children with parental guidance",
    "UA 7+": "UA 7+ — Parental guidance recommended below age 7",
    "UA 13+": "UA 13+ — Parental guidance recommended below age 13",
    "UA 16+": "UA 16+ — Parental guidance recommended below age 16",

    "G": "G — General audiences, suitable for all ages",
    "PG": "PG — Parental guidance suggested",
    "PG-7": "PG-7 — Parental guidance suggested for younger children",
    "PG-13": "PG-13 — Some material may be unsuitable below 13",
    "PG-16": "PG-16 — Some material may be unsuitable below 16",

    "12": "12 — Generally suitable for ages 12 and above",
    "12A": "12A — Under 12s should watch with an adult",
    "15": "15 — Suitable for ages 15 and above",
    "16": "16 — Suitable for ages 16 and above",

    "18": "18 — Suitable for adults aged 18 and above",
    "A": "A — Adult audience",

    "TV-Y": "TV-Y — Designed for young children",
    "TV-Y7": "TV-Y7 — Designed for children aged 7 and above",
    "TV-G": "TV-G — Suitable for general audiences",
    "TV-PG": "TV-PG — Parental guidance suggested",
    "TV-14": "TV-14 — May be unsuitable below age 14",
    "TV-MA": "TV-MA — Intended for mature audiences",

    "NR": "NR — Not Rated",
    "Not Rated": "Not Rated — No age classification available",
    "Unknown": "Unknown — Age classification unavailable"
}


def explain_age_rating(rating):

    rating = str(rating).strip()

    if rating in AGE_RATING_INFO:
        return AGE_RATING_INFO[rating]

    # General automatic explanations
    rating_upper = rating.upper()

    if "18" in rating_upper:
        return f"{rating} — Generally intended for ages 18 and above"

    if "16" in rating_upper:
        return f"{rating} — Generally intended for ages 16 and above"

    if "15" in rating_upper:
        return f"{rating} — Generally intended for ages 15 and above"

    if "13" in rating_upper:
        return f"{rating} — Parental guidance recommended below age 13"

    if "12" in rating_upper:
        return f"{rating} — Generally intended for ages 12 and above"

    if "7" in rating_upper:
        return f"{rating} — Generally intended for ages 7 and above"

    return f"{rating} — Check the applicable classification"


# ============================================================
# AMAZON SEARCH LINK
# ============================================================

def create_amazon_link(title):

    """
    Creates an Amazon search URL for the selected title.

    We deliberately do NOT construct a fake direct Prime
    streaming URL because such URLs can become invalid.

    The user can then select the matching title from Amazon.
    """

    if pd.isna(title):
        return None

    title = str(title).strip()

    if not title:
        return None

    query = urllib.parse.quote_plus(
        f"{title} Amazon Prime"
    )

    return (
        "https://www.amazon.com/s?"
        f"k={query}"
    )


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_movies(
    genre,
    origin,
    language,
    category,
    age_rating
):

    result = data.copy()

    # --------------------------------------------------------
    # Match Score
    # --------------------------------------------------------

    result["Match Score"] = 0

    # --------------------------------------------------------
    # Genre
    # --------------------------------------------------------

    if genre != "Any":

        mask = result["Genre List"].apply(
            lambda x: genre in x
        )

        result.loc[
            mask,
            "Match Score"
        ] += 5

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    if origin != "Any":

        mask = (
            result["Country of origin"]
            == origin
        )

        result.loc[
            mask,
            "Match Score"
        ] += 4

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    if language != "Any":

        mask = (
            result["Language"]
            == language
        )

        result.loc[
            mask,
            "Match Score"
        ] += 4

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    if category != "Any":

        mask = (
            result["Content Category"]
            == category
        )

        result.loc[
            mask,
            "Match Score"
        ] += 3

    # --------------------------------------------------------
    # Age Rating
    # --------------------------------------------------------

    if age_rating != "Any":

        mask = (
            result["Age Rating"]
            == age_rating
        )

        result.loc[
            mask,
            "Match Score"
        ] += 3

    # --------------------------------------------------------
    # IMDb
    # --------------------------------------------------------

    result["IMDb Score"] = pd.to_numeric(
        result["IMDb Score"],
        errors="coerce"
    ).fillna(0)

    result["Number of IMDb Votes"] = pd.to_numeric(
        result["Number of IMDb Votes"],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    result = result.sort_values(
        by=[
            "Match Score",
            "IMDb Score",
            "Number of IMDb Votes"
        ],
        ascending=False
    )

    # --------------------------------------------------------
    # Match explanation
    # --------------------------------------------------------

    def get_match_reason(row):

        reasons = []

        if (
            genre != "Any"
            and genre in row["Genre List"]
        ):
            reasons.append("Genre")

        if (
            origin != "Any"
            and row["Country of origin"] == origin
        ):
            reasons.append("Origin")

        if (
            language != "Any"
            and row["Language"] == language
        ):
            reasons.append("Language")

        if (
            category != "Any"
            and row["Content Category"] == category
        ):
            reasons.append("Category")

        if (
            age_rating != "Any"
            and row["Age Rating"] == age_rating
        ):
            reasons.append("Age Rating")

        return (
            " + ".join(reasons)
            if reasons
            else "IMDb Ranking"
        )

    result["Matched Preferences"] = result.apply(
        get_match_reason,
        axis=1
    )

    # --------------------------------------------------------
    # Top 20
    # --------------------------------------------------------

    result = result.head(20).copy()

    # --------------------------------------------------------
    # Age explanation
    # --------------------------------------------------------

    result["Age Guidance"] = (
        result["Age Rating"]
        .apply(explain_age_rating)
    )

    # --------------------------------------------------------
    # Amazon search URL
    # --------------------------------------------------------

    result["Amazon Link"] = (
        result["Title"]
        .apply(create_amazon_link)
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    output = result[
        [
            "Title",
            "Type",
            "Genre(s)",
            "Language",
            "Country of origin",
            "Age Rating",
            "Age Guidance",
            "Content Category",
            "IMDb Score",
            "Release Year",
            "Duration",
            "Match Score",
            "Matched Preferences",
            "Amazon Link"
        ]
    ].copy()

    output.columns = [
        "Title",
        "Type",
        "Genre",
        "Language",
        "Origin",
        "Age Rating",
        "Age Guidance",
        "Category",
        "IMDb Score",
        "Release Year",
        "Duration",
        "Match Score",
        "Matched Preferences",
        "Amazon Link"
    ]

    return output
    # ============================================================
# STEP 4
# MOVIE MANIA COMPLETE DASHBOARD
# ============================================================


# ============================================================
# FILTER DATA
# ============================================================

def filter_data(
    genre,
    origin,
    language,
    category,
    age_rating
):

    filtered = data.copy()

    if genre != "Any":
        filtered = filtered[
            filtered["Genre List"].apply(
                lambda x: genre in x
            )
        ]

    if origin != "Any":
        filtered = filtered[
            filtered["Country of origin"] == origin
        ]

    if language != "Any":
        filtered = filtered[
            filtered["Language"] == language
        ]

    if category != "Any":
        filtered = filtered[
            filtered["Content Category"] == category
        ]

    if age_rating != "Any":
        filtered = filtered[
            filtered["Age Rating"] == age_rating
        ]

    return filtered


# ============================================================
# KPI
# ============================================================

def create_kpis(
    genre,
    origin,
    language,
    category,
    age_rating
):

    filtered = filter_data(
        genre,
        origin,
        language,
        category,
        age_rating
    )

    total = len(filtered)

    movies = len(
        filtered[
            filtered["Type"].str.lower() == "movie"
        ]
    )

    shows = len(
        filtered[
            filtered["Type"].str.lower() == "tv show"
        ]
    )

    avg_imdb = (
        filtered["IMDb Score"].mean()
        if total > 0
        else 0
    )

    available = int(
        filtered["Is Available"].sum()
    )

    return (

        f"""
        <div class="kpi-card">
            <div class="kpi-number">{total}</div>
            <div class="kpi-label">FILTERED TITLES</div>
        </div>
        """,

        f"""
        <div class="kpi-card">
            <div class="kpi-number">{movies}</div>
            <div class="kpi-label">MOVIES</div>
        </div>
        """,

        f"""
        <div class="kpi-card">
            <div class="kpi-number">{shows}</div>
            <div class="kpi-label">TV SHOWS</div>
        </div>
        """,

        f"""
        <div class="kpi-card">
            <div class="kpi-number">{avg_imdb:.2f}</div>
            <div class="kpi-label">AVG IMDb</div>
        </div>
        """,

        f"""
        <div class="kpi-card">
            <div class="kpi-number">{available}</div>
            <div class="kpi-label">AVAILABLE</div>
        </div>
        """
    )


# ============================================================
# CHART FUNCTION
# ============================================================

def make_charts(filtered, title_prefix=""):

    # ========================================================
    # 1. MOVIE TYPE
    # ========================================================

    type_counts = (
        filtered["Type"]
        .value_counts()
        .reset_index()
    )

    type_counts.columns = [
        "Type",
        "Number"
    ]

    fig_type = px.pie(
        type_counts,
        names="Type",
        values="Number",
        hole=0.45,
        title=f"{title_prefix} Movie Type"
    )

    fig_type.update_layout(
        template="plotly_dark",
        height=450
    )


    # ========================================================
    # 2. COUNTRY OF ORIGIN
    # ========================================================

    country_counts = (
        filtered["Country of origin"]
        .value_counts()
        .reset_index()
    )

    country_counts.columns = [
        "Country",
        "Number"
    ]

    country_counts = country_counts.head(20)

    fig_country = px.bar(
        country_counts.sort_values("Number"),
        x="Number",
        y="Country",
        orientation="h",
        title=f"{title_prefix} Country of Origin",
        text="Number"
    )

    fig_country.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_title="Number of Titles",
        yaxis_title="Country"
    )


    # ========================================================
    # 3. MOVIES / SHOWS PER YEAR
    # ========================================================

    yearly = (
        filtered
        .groupby(
            ["Release Year", "Type"]
        )
        .size()
        .reset_index(
            name="Number"
        )
    )

    yearly = yearly.dropna(
        subset=["Release Year"]
    )

    yearly["Release Year"] = (
        yearly["Release Year"]
        .astype(int)
    )

    fig_yearly = px.line(
        yearly,
        x="Release Year",
        y="Number",
        color="Type",
        markers=True,
        title=f"{title_prefix} Movies & TV Shows per Year"
    )

    fig_yearly.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )


    # ========================================================
    # 4. GENRE × LANGUAGE
    # ========================================================

    gl = filtered.explode(
        "Genre List"
    ).copy()

    gl["Genre"] = (
        gl["Genre List"]
        .fillna("Unknown")
    )

    gl = (
        gl.groupby(
            ["Genre", "Language"]
        )
        .size()
        .reset_index(
            name="Number"
        )
    )

    top_genres = (
        gl.groupby("Genre")["Number"]
        .sum()
        .nlargest(15)
        .index
    )

    gl = gl[
        gl["Genre"].isin(top_genres)
    ]

    fig_genre_language = px.density_heatmap(
        gl,
        x="Language",
        y="Genre",
        z="Number",
        histfunc="sum",
        text_auto=True,
        title=f"{title_prefix} Genre × Language"
    )

    fig_genre_language.update_layout(
        template="plotly_dark",
        height=650
    )


    # ========================================================
    # 5. TV SHOWS PER YEAR
    # ========================================================

    tv = filtered[
        filtered["Type"].str.lower()
        == "tv show"
    ].copy()

    tv_year = (
        tv.groupby("Release Year")
        .size()
        .reset_index(
            name="Number of TV Shows"
        )
    )

    tv_year = tv_year.dropna(
        subset=["Release Year"]
    )

    tv_year["Release Year"] = (
        tv_year["Release Year"]
        .astype(int)
    )

    fig_tv_year = px.bar(
        tv_year,
        x="Release Year",
        y="Number of TV Shows",
        title=f"{title_prefix} TV Shows per Year",
        text="Number of TV Shows"
    )

    fig_tv_year.update_layout(
        template="plotly_dark",
        height=500
    )


    # ========================================================
    # 6. TV SHOWS BY GENRE
    # ========================================================

    tv_genre = tv.explode(
        "Genre List"
    ).copy()

    tv_genre["Genre"] = (
        tv_genre["Genre List"]
        .fillna("Unknown")
    )

    tv_genre_counts = (
        tv_genre["Genre"]
        .value_counts()
        .reset_index()
    )

    tv_genre_counts.columns = [
        "Genre",
        "Number of TV Shows"
    ]

    tv_genre_counts = (
        tv_genre_counts
        .head(15)
    )

    fig_tv_genre = px.bar(
        tv_genre_counts.sort_values(
            "Number of TV Shows"
        ),
        x="Number of TV Shows",
        y="Genre",
        orientation="h",
        title=f"{title_prefix} TV Shows by Genre",
        text="Number of TV Shows"
    )

    fig_tv_genre.update_layout(
        template="plotly_dark",
        height=600
    )


    return (
        fig_type,
        fig_country,
        fig_yearly,
        fig_genre_language,
        fig_tv_year,
        fig_tv_genre
    )


# ============================================================
# OVERALL EDA
# ============================================================

def create_overall_charts():

    return make_charts(
        data,
        "🌐 Overall Dataset —"
    )


# ============================================================
# FILTERED EDA
# ============================================================

def create_filtered_charts(
    genre,
    origin,
    language,
    category,
    age_rating
):

    filtered = filter_data(
        genre,
        origin,
        language,
        category,
        age_rating
    )

    return make_charts(
        filtered,
        "🎯 Filtered Dataset —"
    )


# ============================================================
# UPDATE DASHBOARD
# ============================================================

def update_dashboard(
    genre,
    origin,
    language,
    category,
    age_rating
):

    # KPIs
    kpis = create_kpis(
        genre,
        origin,
        language,
        category,
        age_rating
    )

    # Filtered charts
    filtered_charts = create_filtered_charts(
        genre,
        origin,
        language,
        category,
        age_rating
    )

    # Recommendations
    recommendations = recommend_movies(
        genre,
        origin,
        language,
        category,
        age_rating
    )

    titles = recommendations[
        "Title"
    ].tolist()

    return (
        *kpis,
        *filtered_charts,
        recommendations,
        gr.update(
            choices=titles,
            value=None
        )
    )


# ============================================================
# WATCH SELECTED TITLE
# ============================================================

def selected_title_link(
    title,
    genre,
    origin,
    language,
    category,
    age_rating
):

    if not title:

        return """
        <div class="watch-box">
            <h3>🎬 Select a title</h3>
            <p>
                Select one of the recommended titles
                to open its Amazon search page.
            </p>
        </div>
        """

    recommendations = recommend_movies(
        genre,
        origin,
        language,
        category,
        age_rating
    )

    row = recommendations[
        recommendations["Title"] == title
    ]

    if row.empty:

        return """
        <div class="watch-box">
            <h3>Title not found</h3>
        </div>
        """

    row = row.iloc[0]

    amazon_url = row["Amazon Link"]

    age_text = row["Age Guidance"]

    return f"""
    <div class="watch-box">

        <h2>🎬 {row["Title"]}</h2>

        <p>
            <b>Type:</b> {row["Type"]}
            &nbsp;&nbsp; | &nbsp;&nbsp;
            <b>Genre:</b> {row["Genre"]}
        </p>

        <p>
            <b>Language:</b> {row["Language"]}
            &nbsp;&nbsp; | &nbsp;&nbsp;
            <b>IMDb:</b> {row["IMDb Score"]}
        </p>

        <p>
            <b>Age Rating:</b> {row["Age Rating"]}
        </p>

        <p>
            {age_text}
        </p>

        <div class="watch-button">

            <a
                href="{amazon_url}"
                target="_blank"
            >
                🔎 Find "{row["Title"]}" on Amazon
            </a>

        </div>

        <p class="small-note">
            Amazon availability can vary by country and title.
        </p>

    </div>
    """


# ============================================================
# PURPLE / BLACK CSS
# ============================================================

custom_css = """

/* =========================================================
   COMPLETE PAGE
   ========================================================= */

body {
    background:
        radial-gradient(
            circle at top left,
            #32105f 0%,
            #170b2b 35%,
            #050509 100%
        ) !important;

    color: white !important;
}


/* Main Gradio container */

.gradio-container {
    background:
        linear-gradient(
            135deg,
            #09050f,
            #180b2c,
            #0a0610
        ) !important;

    color: white !important;
}


/* =========================================================
   HEADINGS
   ========================================================= */

h1 {
    color: #d8b4fe !important;
}

h2 {
    color: #c084fc !important;
}

h3 {
    color: #e9d5ff !important;
}

#main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 900;

    color: #e9d5ff !important;

    text-shadow:
        0 0 15px #7e22ce,
        0 0 30px #581c87;
}

#subtitle {
    text-align: center;
    color: #c4b5fd !important;
    font-size: 18px;
    margin-bottom: 30px;
}


/* =========================================================
   KPI CARDS
   ========================================================= */

.kpi-card {
    background:
        linear-gradient(
            145deg,
            #24103d,
            #100817
        );

    border:
        1px solid #6b21a8;

    border-radius: 18px;

    padding: 20px;

    text-align: center;

    box-shadow:
        0 0 15px rgba(126,34,206,0.25);
}

.kpi-number {
    color: #e9d5ff;
    font-size: 30px;
    font-weight: 900;
}

.kpi-label {
    color: #c4b5fd;
    font-size: 13px;
}


/* =========================================================
   WATCH BOX
   ========================================================= */

.watch-box {

    background:
        linear-gradient(
            145deg,
            #24103d,
            #0d0715
        );

    border:
        1px solid #7e22ce;

    border-radius: 20px;

    padding: 30px;

    text-align: center;

    margin-top: 15px;
    margin-bottom: 25px;

    box-shadow:
        0 0 25px rgba(126,34,206,0.25);
}

.watch-box h2 {
    color: #e9d5ff !important;
}

.watch-box p {
    color: #ddd6fe !important;
}

.watch-button {
    margin-top: 25px;
}

.watch-button a {

    display: inline-block;

    padding: 14px 28px;

    border-radius: 12px;

    text-decoration: none;

    font-weight: 800;

    color: white !important;

    background:
        linear-gradient(
            90deg,
            #6b21a8,
            #9333ea
        );

    box-shadow:
        0 0 20px rgba(147,51,234,0.4);
}

.watch-button a:hover {

    background:
        linear-gradient(
            90deg,
            #9333ea,
            #c084fc
        );
}

.small-note {
    margin-top: 15px;
    font-size: 12px;
    color: #a78bfa !important;
}


/* =========================================================
   DROPDOWNS
   ========================================================= */

label {
    color: #ddd6fe !important;
}

input,
textarea,
select {

    background: #12091d !important;

    color: white !important;

    border:
        1px solid #6b21a8 !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

button {

    background:
        linear-gradient(
            90deg,
            #6b21a8,
            #9333ea
        ) !important;

    color: white !important;

    border: none !important;
}

button:hover {

    background:
        linear-gradient(
            90deg,
            #9333ea,
            #c084fc
        ) !important;
}


/* =========================================================
   DATAFRAME
   ========================================================= */

.dataframe {

    background: #100817 !important;

    color: white !important;
}


/* =========================================================
   MARKDOWN
   ========================================================= */

.markdown-text {

    color: #ddd6fe !important;
}


/* =========================================================
   PLOTLY
   ========================================================= */

.plot-container {

    border:
        1px solid #3b1764;

    border-radius: 15px;

    overflow: hidden;
}

"""


# ============================================================
# BUILD APPLICATION
# ============================================================

with gr.Blocks(
    theme=gr.themes.Base(),
    css=custom_css,
    title="Movie Mania"
) as app:

    # ========================================================
    # HEADER
    # ========================================================

    gr.Markdown(
        "# 🎬 MOVIE MANIA",
        elem_id="main-title"
    )

    gr.Markdown(
        "Amazon Prime Movies & TV Shows "
        "Interactive EDA & Recommendation Dashboard",
        elem_id="subtitle"
    )


    # ========================================================
    # FILTER SECTION
    # ========================================================

    gr.Markdown(
        "## 🎯 Find Your Movie or Show"
    )

    with gr.Row():

        genre_input = gr.Dropdown(
            choices=["Any"] + GENRES,
            value="Any",
            label="🎭 Genre"
        )

        origin_input = gr.Dropdown(
            choices=["Any"] + ORIGINS,
            value="Any",
            label="🌍 Country of Origin"
        )

        language_input = gr.Dropdown(
            choices=["Any"] + LANGUAGES,
            value="Any",
            label="🗣️ Language"
        )

    with gr.Row():

        category_input = gr.Dropdown(
            choices=["Any"] + CATEGORIES,
            value="Any",
            label="🎬 Content Category"
        )

        rating_input = gr.Dropdown(
            choices=["Any"] + AGE_RATINGS,
            value="Any",
            label="🔞 Age Rating"
        )

    recommend_button = gr.Button(
        "🔍 RECOMMEND",
        variant="primary"
    )


    # ========================================================
    # FILTERED KPI
    # ========================================================

    gr.Markdown(
        "## 📊 Selected Preferences Overview"
    )

    with gr.Row():

        kpi_total = gr.HTML()

        kpi_movies = gr.HTML()

        kpi_shows = gr.HTML()

        kpi_score = gr.HTML()

        kpi_available = gr.HTML()


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    gr.Markdown(
        "## 🍿 Recommended Movies & Shows"
    )

    recommendation_table = gr.Dataframe(
        interactive=False,
        wrap=True
    )


    # ========================================================
    # SELECT TITLE
    # ========================================================

    gr.Markdown(
        "## ▶️ Find Selected Title on Amazon"
    )

    title_selector = gr.Dropdown(
        choices=[],
        label="Select a recommended title",
        interactive=True
    )

    watch_result = gr.HTML(
        value="""
        <div class="watch-box">
            <h3>🎬 Select a title</h3>
            <p>
                Select a recommended title to find it on Amazon.
            </p>
        </div>
        """
    )


    # ========================================================
    # ========================================================
    # OVERALL EDA
    # ========================================================
    # ========================================================

    gr.Markdown("---")

    gr.Markdown(
        "# 🌐 Overall Dataset EDA"
    )

    gr.Markdown(
        """
        These graphs use the **complete dataset**.

        They are independent of the recommendation filters,
        so selecting a genre, language, country or rating
        will NOT change these graphs.
        """
    )


    overall_charts = create_overall_charts()

    # Overall graph 1 + 2

    with gr.Row():

        gr.Plot(
            value=overall_charts[0],
            label="Overall Movie Type"
        )

        gr.Plot(
            value=overall_charts[1],
            label="Overall Country of Origin"
        )


    # Overall graph 3

    gr.Plot(
        value=overall_charts[2],
        label="Overall Movies & Shows per Year"
    )


    # Overall graph 4

    gr.Plot(
        value=overall_charts[3],
        label="Overall Genre × Language"
    )


    # Overall graph 5 + 6

    with gr.Row():

        gr.Plot(
            value=overall_charts[4],
            label="Overall TV Shows per Year"
        )

        gr.Plot(
            value=overall_charts[5],
            label="Overall TV Shows by Genre"
        )


    # ========================================================
    # ========================================================
    # FILTERED EDA
    # ========================================================
    # ========================================================

    gr.Markdown("---")

    gr.Markdown(
        "# 🎯 Filtered EDA"
    )

    gr.Markdown(
        """
        These graphs respond to the preferences selected above.
        """
    )


    graph_type = gr.Plot(
        label="Filtered Movie Type"
    )

    graph_country = gr.Plot(
        label="Filtered Country of Origin"
    )

    graph_yearly = gr.Plot(
        label="Filtered Movies & TV Shows per Year"
    )

    graph_genre_language = gr.Plot(
        label="Filtered Genre × Language"
    )

    with gr.Row():

        graph_tv_year = gr.Plot(
            label="Filtered TV Shows per Year"
        )

        graph_tv_genre = gr.Plot(
            label="Filtered TV Shows by Genre"
        )


    # ========================================================
    # AGE RATING REFERENCE
    # ========================================================

    gr.Markdown("---")

    gr.Markdown(
        "# 🔞 Age Rating Guide"
    )

    age_guide_data = pd.DataFrame({

        "Code": [
            "U",
            "UA",
            "UA 7+",
            "UA 13+",
            "UA 16+",
            "G",
            "PG",
            "PG-13",
            "12",
            "12A",
            "15",
            "16",
            "18",
            "A",
            "TV-Y",
            "TV-Y7",
            "TV-G",
            "TV-PG",
            "TV-14",
            "TV-MA",
            "NR"
        ],

        "Age Guidance": [
            "All ages",
            "Children — parental guidance",
            "Parental guidance below 7",
            "Parental guidance below 13",
            "Parental guidance below 16",
            "General audiences",
            "Parental guidance",
            "May be unsuitable below 13",
            "Generally 12+",
            "Under 12 with adult",
            "15+",
            "16+",
            "18+",
            "Adult audience",
            "Young children",
            "Children 7+",
            "General audiences",
            "Parental guidance",
            "May be unsuitable below 14",
            "Mature audiences",
            "Not rated"
        ]
    })

    gr.Dataframe(
        value=age_guide_data,
        interactive=False,
        label="Age Rating Code Reference"
    )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.Markdown(
        """
        ---

        ## 🎬 Movie Mania

        **Interactive EDA + Preference-Based Recommendation**

        Overall EDA → User Preferences → Recommendations
        → Amazon Search → Filtered EDA
        """
    )


    # ========================================================
    # INPUTS
    # ========================================================

    inputs = [
        genre_input,
        origin_input,
        language_input,
        category_input,
        rating_input
    ]


    # ========================================================
    # OUTPUTS
    # ========================================================

    outputs = [

        # KPIs
        kpi_total,
        kpi_movies,
        kpi_shows,
        kpi_score,
        kpi_available,

        # Filtered EDA
        graph_type,
        graph_country,
        graph_yearly,
        graph_genre_language,
        graph_tv_year,
        graph_tv_genre,

        # Recommendations
        recommendation_table,

        # Title selector
        title_selector
    ]


    # ========================================================
    # RECOMMEND BUTTON
    # ========================================================

    recommend_button.click(
        fn=update_dashboard,
        inputs=inputs,
        outputs=outputs
    )


    # ========================================================
    # AUTOMATIC FILTER UPDATE
    # ========================================================

    for component in inputs:

        component.change(
            fn=update_dashboard,
            inputs=inputs,
            outputs=outputs
        )


    # ========================================================
    # AMAZON TITLE SELECTION
    # ========================================================

    title_selector.change(
        fn=selected_title_link,
        inputs=[
            title_selector,
            genre_input,
            origin_input,
            language_input,
            category_input,
            rating_input
        ],
        outputs=watch_result
    )


# ============================================================
# INITIAL DASHBOARD
# ============================================================

initial = update_dashboard(
    "Any",
    "Any",
    "Any",
    "Any",
    "Any"
)

for component, value in zip(
    outputs,
    initial
):

    component.value = value


print("🎬 Movie Mania dashboard created successfully!")
# ============================================================
# STEP 5 — RENDER LAUNCH
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )
