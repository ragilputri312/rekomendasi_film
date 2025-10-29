import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="🎬 Sistem Rekomendasi Film Berdasarkan Aktor",
    layout="wide",
)

# ===============================
# STYLING (CSS)
# ===============================
st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #2b2d42;
    color: white;
}
.section {
    background-color: #edf2f4;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
footer {
    text-align: center;
    color: gray;
    font-size: 0.85em;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# NAVBAR DI SEBELAH KIRI
# ===============================
st.sidebar.title("🎬 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Tentang Aplikasi", "🎬 Cari Rekomendasi Film", "📊 Dataset & Visualisasi"]
)

# ===============================
# LOAD DATASET
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("actorfilms.csv")
    df.dropna(subset=["Actor", "Film", "Rating"], inplace=True)
    df["Rating"] = df["Rating"].astype(float)
    return df

df = load_data()

# ===============================
# HALAMAN 1 — TENTANG APLIKASI
# ===============================
if menu == "🏠 Tentang Aplikasi":
    st.title("🎬 Sistem Rekomendasi Film Berdasarkan Aktor")
    st.markdown("""
    <div class="section">
    Sistem ini menggunakan metode Content-Based Filtering untuk memberikan rekomendasi film
    berdasarkan kesamaan filmografi antar aktor.  
    <br><br>
    💡 Cara kerja singkat:<br>
    - Setiap aktor direpresentasikan oleh kumpulan film yang pernah ia mainkan. <br>
    - Dengan menggunakan TF-IDF Vectorizer, sistem mengubah daftar film menjadi vektor teks. <br>
    - Kemudian dihitung cosine similarity antar aktor untuk menemukan aktor dengan filmografi serupa. <br>
    - Film dari aktor yang paling mirip dijadikan rekomendasi. 
    <br><br>
    🎯 Tujuan: <br>
    Membantu pengguna menemukan film baru yang relevan dengan preferensi aktor favoritnya.
    </div>
    """, unsafe_allow_html=True)

    st.success("Coba buka menu 🎬 *Cari Rekomendasi Film* untuk mulai eksplorasi!")

# ===============================
# HALAMAN 2 — REKOMENDASI FILM
# ===============================
elif menu == "🎬 Cari Rekomendasi Film":
    st.header("🎬 Cari Rekomendasi Film")
    st.write("Temukan film serupa berdasarkan aktor favoritmu 🎭")

    actor_list = sorted(df["Actor"].unique().tolist())
    selected_actor = st.selectbox(
        "🔍 Pilih atau ketik nama aktor:",
        actor_list,
        index=None,
        placeholder="Ketik nama aktor..."
    )

    if selected_actor:
        # --- Filmografi ---
        actor_films = df[df["Actor"] == selected_actor]["Film"].tolist()
        st.markdown(f"### 🎞️ Filmografi {selected_actor}")
        st.markdown("\n".join([f"- 🎬 {film}" for film in actor_films[:15]]))

        # --- Similarity antar aktor ---
        actor_group = df.groupby("Actor")["Film"].apply(lambda x: " ".join(x)).reset_index()
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(actor_group["Film"])
        similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
        sim_df = pd.DataFrame(similarity, index=actor_group["Actor"], columns=actor_group["Actor"])

        similar_actors = sim_df[selected_actor].sort_values(ascending=False)[1:6]

        # --- Aktor Serupa ---
        st.markdown("### 👥 Aktor dengan Filmografi Serupa")
        for i, (actor, score) in enumerate(similar_actors.items(), start=1):
            st.markdown(f"**{i}. {actor}** — Skor kemiripan: `{score:.3f}`")

        # --- Rekomendasi Film ---
        st.markdown("### 🍿 Rekomendasi Film dari Aktor Serupa")
        similar_actors_list = similar_actors.index.tolist()
        recommended_films = (
            df[df["Actor"].isin(similar_actors_list)]
            .sort_values(by="Rating", ascending=False)
            .drop_duplicates(subset=["Film"])
            .head(10)
        )

        st.dataframe(
            recommended_films[["Film", "Actor", "Year", "Rating", "Votes"]],
            use_container_width=True
        )

    else:
        st.info("🔎 Silakan pilih aktor terlebih dahulu untuk melihat rekomendasi film.")

# ===============================
# HALAMAN 3 — DATASET & VISUALISASI
# ===============================
elif menu == "📊 Dataset & Visualisasi":
    st.header("📊 Dataset dan Visualisasi")

    tab1, tab2 = st.tabs(["🎥 Data Film", "🧑‍🎤 Data Aktor"])

    # --- TAB 1: Data Film ---
    with tab1:
        st.subheader("🎥 Data Film")
        film_data = df[["Film", "Year", "Rating", "Votes"]].drop_duplicates()
        st.dataframe(film_data, use_container_width=True)

        # Grafik distribusi rating
        fig = px.histogram(df, x="Rating", nbins=20, title="Distribusi Rating Film", color_discrete_sequence=["#2b2d42"])
        st.plotly_chart(fig, use_container_width=True)

        # Grafik jumlah film per tahun
        fig2 = px.bar(df.groupby("Year").size().reset_index(name="Jumlah Film"),
                      x="Year", y="Jumlah Film", title="Jumlah Film per Tahun", color_discrete_sequence=["#ef233c"])
        st.plotly_chart(fig2, use_container_width=True)

        # --- Tambahan: Film dengan Rating Tertinggi ---
        st.subheader("🏆 Film dengan Rating Tertinggi")
        top_films = film_data.sort_values(by="Rating", ascending=False).head(10)
        fig3 = px.bar(top_films, x="Rating", y="Film", orientation="h",
                      title="Top 10 Film dengan Rating Tertinggi", color="Rating",
                      color_continuous_scale="Viridis")
        st.plotly_chart(fig3, use_container_width=True)

        # --- Lihat Detail Film ---
        st.markdown("---")
        st.subheader("🔍 Lihat Detail Film")
        selected_film = st.selectbox("Pilih film untuk melihat aktor yang berperan:", sorted(df["Film"].unique()))
        if selected_film:
            film_detail = df[df["Film"] == selected_film][["Actor", "Rating"]].sort_values(by="Rating", ascending=False)
            st.markdown(f"### 🎬 Film: {selected_film}")
            st.dataframe(film_detail, use_container_width=True)

    # --- TAB 2: Data Aktor ---
    with tab2:
        st.subheader("🧑‍🎤 Data Aktor")
        actor_summary = df.groupby("Actor").agg({
            "Film": "count",
            "Rating": "mean",
            "Votes": "sum"
        }).rename(columns={"Film": "Jumlah Film", "Rating": "Rata-rata Rating", "Votes": "Total Votes"}).reset_index()

        st.dataframe(actor_summary, use_container_width=True)

        # Top 10 aktor dengan rating rata-rata tertinggi
        top_actors = actor_summary.sort_values(by="Rata-rata Rating", ascending=False).head(10)
        fig4 = px.bar(top_actors, x="Rata-rata Rating", y="Actor", orientation="h",
                      title="Top 10 Aktor dengan Rata-rata Rating Tertinggi", color="Rata-rata Rating",
                      color_continuous_scale="RdBu")
        st.plotly_chart(fig4, use_container_width=True)

        # --- Lihat Detail Aktor ---
        st.markdown("---")
        st.subheader("🔍 Lihat Detail Aktor")
        selected_actor = st.selectbox("Pilih nama aktor untuk melihat film yang dibintanginya:", sorted(df["Actor"].unique()))
        if selected_actor:
            actor_detail = df[df["Actor"] == selected_actor][["Film", "Year", "Rating"]].drop_duplicates()
            st.markdown(f"### 🎭 Aktor: {selected_actor}")
            st.dataframe(actor_detail, use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<footer>
<hr>
📍 Dibuat dengan ❤️ oleh <b>Kelompok - Kinder Joy</b> — Sistem Pemberi Rekomendasi 2025
</footer>
""", unsafe_allow_html=True)
