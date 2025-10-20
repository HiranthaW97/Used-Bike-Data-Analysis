import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------------
# Streamlit Page Config
# -----------------------------------

st.set_page_config(
    page_title="Sri Lanka Used Bike Price Advisor",
    layout="wide",
    page_icon="🏍️"
)

st.markdown(
    "<h1 style='text-align: center;'>🏍️ Sri Lanka Used Bike Market — Insights Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <p style='text-align: center; font-size: 18px;'>
    This dashboard analyzes <b>used bike prices in Sri Lanka</b>  
    to help buyers, sellers, and dealers understand market trends and estimate fair prices.
    </p>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# Load Exported Analysis Data (from notebook exports)
# -----------------------------------
avg_year = pd.read_csv("exports/avg_price_by_year.csv")
avg_brand = pd.read_csv("exports/avg_price_by_brand.csv")
bike_types = pd.read_csv("exports/bike_type_counts.csv")
avg_city = pd.read_csv("exports/avg_price_by_city.csv")
dep_brand = pd.read_csv("exports/depreciation_by_brand.csv")
city_demand = pd.read_csv("exports/city_demand_vs_price.csv")

# -----------------------------------
# Load Pre-trained Model (.pkl from Colab)
# -----------------------------------
model = joblib.load("model/bike_price_model.pkl")

# -----------------------------------
# Tabs
# -----------------------------------
analytics_tab, forecast_tab,reference_tab = st.tabs(["📊 Market Analytics", "🔮 Forecast Price","📂 Reference Data"])

# -----------------------------------
# Tab 1: Market Analytics
# -----------------------------------
with analytics_tab:
    st.markdown(
        "<h2 style='text-align: center;'>📊 Market Analytics</h2>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 Depreciation Trend — Avg Price by Year")
        fig_year = px.line(
            avg_year,
            x="Year",
            y="avg_price",
            markers=True,
            template="plotly_dark",
            title="Average Price by Year"
        )
        st.plotly_chart(fig_year, use_container_width=True)

    with col2:
        st.subheader("🏍️ Top Brands by Average Price")
        fig_brand = px.bar(
            avg_brand.sort_values("avg_price", ascending=False).head(10),
            x="Brand",
            y="avg_price",
            text="avg_price",
            color="avg_price",
            color_continuous_scale="Blues",
            template="plotly_white"
        )
        fig_brand.update_traces(texttemplate="Rs. %{text:,.0f}", textposition="outside")
        st.plotly_chart(fig_brand, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📊 Bike Types by Demand")
        fig_type = px.pie(
            bike_types,
            names="BikeType",
            values="n",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_type.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_type, use_container_width=True)

    with col4:
        st.subheader("🌍 Top Cities by Avg Price")
        fig_city = px.bar(
            avg_city.sort_values("avg_price", ascending=False).head(10),
            x="City",
            y="avg_price",
            color="avg_price",
            text="avg_price",
            color_continuous_scale="Oranges",
            template="plotly_white"
        )
        fig_city.update_traces(texttemplate="Rs. %{text:,.0f}", textposition="outside")
        st.plotly_chart(fig_city, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("📉 Depreciation by Brand (Retention Ratio)")
        fig_dep = px.bar(
            dep_brand.sort_values("retention_ratio", ascending=False).head(10),
            x="Brand",
            y="retention_ratio",
            color="retention_ratio",
            text="retention_ratio",
            color_continuous_scale="Viridis",
            template="plotly_white"
        )
        fig_dep.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig_dep, use_container_width=True)

    with col6:
        st.subheader("📈 City Demand vs Price")
        fig_demand = px.scatter(
            city_demand,
            x="listings",
            y="avg_price",
            size="listings",
            color="avg_price",
            hover_name="City",
            template="plotly_white",
            title="City Demand vs Average Price"
        )
        st.plotly_chart(fig_demand, use_container_width=True)

# -----------------------------------
# Tab 2: Forecasting & Prediction
# -----------------------------------
with forecast_tab:
    st.markdown("Enter bike details to estimate its **fair market price**:")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.text_input("Brand", "Honda")
        bike_type = st.text_input("Bike Type", "Scooter")
        model_name = st.text_input("Model", "Dio")
    with col2:
        year = st.number_input("Year Registered", min_value=1990, max_value=2025, value=2019)
        mileage = st.number_input("Mileage (km)", min_value=0, value=25000)

    if st.button("Check Price"):
        # Build input DataFrame with correct schema
        user_input = pd.DataFrame([{
            "Brand": brand,
            "Bike Type": bike_type,
            "Model": model_name,
            "Year": int(year),
            "Mileage": int(mileage)
        }])

        # Predict with sklearn pipeline
        prediction = model.predict(user_input)[0]

        st.success(f"💰 Estimated Market Price: Rs. {prediction:,.0f}")


with reference_tab:
    #st.caption("📌 Data source: **ikman.lk scraped dataset**")

    st.caption(
    "📌 Data source: [**ikman.lk scraped dataset**](https://www.kaggle.com/datasets/ravinathwanni/used-bike-prices-in-sri-lanka)"
    )

    df = pd.read_csv("data/used-bikes.csv")
    st.dataframe(df.head(50))  # show first 50 rows
    
    st.download_button(
        "⬇️ Download full dataset (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "used_bikes.csv",
        "text/csv"
    )
