import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Primetrade AI Dashboard", layout="wide")
st.title("📊 Trader Performance vs Market Sentiment")

try:
    df = pd.read_csv(r"C:\Users\MSII\Desktop\Primetrade_Assignment\enriched_data.csv")

    st.sidebar.header("Filters")
    sentiment_select = st.sidebar.multiselect(
        "Market Sentiment",
        df["Classification"].unique(),
        default=df["Classification"].unique()
    )

    archetype_select = st.sidebar.multiselect(
        "Trader Archetype",
        df["Archetype_Name"].unique(),
        default=df["Archetype_Name"].unique()
    )

    filtered_df = df[
        (df["Classification"].isin(sentiment_select)) &
        (df["Archetype_Name"].isin(archetype_select))
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data Points", len(filtered_df))
    c2.metric("Avg PnL", f"${filtered_df['Closed PnL'].mean():.2f}")
    c3.metric("Win Rate (%)", f"{(filtered_df['Closed PnL'] > 0).mean()*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance by Sentiment")
        fig1 = px.bar(filtered_df, x="Classification", y="Closed PnL", color="Classification")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Trader Archetypes")
        fig2 = px.pie(filtered_df, names="Archetype_Name", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Activity vs Profitability")
    fig3 = px.scatter(
        filtered_df,
        x="trade_frequency",
        y="Closed PnL",
        color="Archetype_Name",
        hover_data=["Account"]
    )
    st.plotly_chart(fig3, use_container_width=True)

except FileNotFoundError:
    st.error("enriched_data.csv file nahi mili")

except Exception as e:
    st.error(e)