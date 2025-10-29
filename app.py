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
# STYLING (CSS) DENGAN BOOTSTRAP ICONS
# ===============================
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
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
.team-card {
    background-color: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1.5rem;
    text-align: center;
    height: 100%;
}
.team-photo {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    object-fit: cover;
    margin: 0 auto 1rem;
    border: 4px solid #2b2d42;
}
.menu-icon {
    margin-right: 8px;
}
.header-icon {
    margin-right: 10px;
}
.bootstrap-icon {
    display: inline-block;
    margin-right: 5px;
}
</style>
""", unsafe_allow_html=True)

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
# NAVBAR DI SEBELAH KIRI
# ===============================
menu_options = [
    "👥 Biodata Kelompok",
    "🏠 Tentang Aplikasi",
    "🔍 Cari Rekomendasi Film",
    "📊 Dataset & Visualisasi",
]
if "menu_selected" not in st.session_state:
    st.session_state["menu_selected"] = menu_options[0]

st.sidebar.markdown('<h2><i class="bi bi-film"></i> Menu Navigasi</h2>', unsafe_allow_html=True)
st.sidebar.markdown("---")

for idx, option in enumerate(menu_options):
    is_active = st.session_state["menu_selected"] == option
    clicked = st.sidebar.button(option, use_container_width=True, disabled=is_active, key=f"menu_btn_{idx}")
    if clicked:
        st.session_state["menu_selected"] = option
        st.rerun()

selected_menu = st.session_state["menu_selected"]

# ===============================
# HALAMAN 1 — BIODATA KELOMPOK
# ===============================
if selected_menu == "👥 Biodata Kelompok":
    st.markdown('<h1><i class="bi bi-people-fill header-icon"></i> Biodata Kelompok Kinder Joy</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section">
    Berikut adalah anggota kelompok <strong>Kinder Joy</strong> yang mengembangkan Sistem Rekomendasi Film Berdasarkan Aktor:
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="team-card">
            <div class="team-photo" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; color: white;">
                <i class="bi bi-person-fill" style="font-size: 3rem;"></i>
            </div>
            <h3></i> Ragil Putri Rahmadani</h3>
            <p></i> NIM: 2211102346</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="team-card">
            <div class="team-photo" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); display: flex; align-items: center; justify-content: center; color: white;">
                <i class="bi bi-person-fill" style="font-size: 3rem;"></i>
            </div>
            <h3></i> M. Rizki Agustiansyah</h3>
            <p></i> NIM: 2211102325</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="team-card">
            <div class="team-photo" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); display: flex; align-items: center; justify-content: center; color: white;">
                <i class="bi bi-person-fill" style="font-size: 3rem;"></i>
            </div>
            <h3></i> Azfa Fairuzia Hartoyo</h3>
            <p></i> NIM: 2211102328</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section">
    <h4><i class="bi bi-info-circle"></i> Tentang Kelompok</h4>
    <p>Kelompok <strong>Kinder Joy</strong> terdiri dari tiga anggota yang bekerja sama dalam mengembangkan sistem rekomendasi film berdasarkan aktor favorit pengguna. Sistem ini menggunakan metode Content-Based Filtering untuk memberikan rekomendasi yang personal dan relevan.</p>
    
    <h4><i class="bi bi-tools"></i> Teknologi yang Digunakan</h4>
    <p><i class="bi bi-check-circle"></i> Python <i class="bi bi-check-circle"></i> Streamlit <i class="bi bi-check-circle"></i> Scikit-learn <i class="bi bi-check-circle"></i> Pandas <i class="bi bi-check-circle"></i> Plotly</p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# HALAMAN 2 — TENTANG APLIKASI
# ===============================
elif selected_menu == "🏠 Tentang Aplikasi":
    st.markdown('<h1><i class="bi bi-house-door-fill header-icon"></i> Sistem Rekomendasi Film Berdasarkan Aktor</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="section">
        <h4><i class="bi bi-lightbulb-fill"></i> Tentang Sistem</h4>
        <p>Sistem ini menggunakan metode <strong>Content-Based Filtering</strong> untuk memberikan rekomendasi film berdasarkan kesamaan filmografi antar aktor.</p>
        
        <h4><i class="bi bi-gear-fill"></i> Cara Kerja</h4>
        <ul>
        <li><i class="bi bi-check-square"></i> Setiap aktor direpresentasikan oleh kumpulan film yang pernah ia mainkan</li>
        <li><i class="bi bi-check-square"></i> Dengan menggunakan <strong>TF-IDF Vectorizer</strong>, sistem mengubah daftar film menjadi vektor teks</li>
        <li><i class="bi bi-check-square"></i> Kemudian dihitung <strong>cosine similarity</strong> antar aktor untuk menemukan aktor dengan filmografi serupa</li>
        <li><i class="bi bi-check-square"></i> Film dari aktor yang paling mirip dijadikan rekomendasi</li>
        </ul>
        
        <h4><i class="bi bi-bullseye"></i> Tujuan</h4>
        <p>Membantu pengguna menemukan film baru yang relevan dengan preferensi aktor favoritnya.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section" style="text-align: center;">
        <h4><i class="bi bi-diagram-3"></i> Alur Sistem</h4>
        <div style="font-size: 0.9em;">
        <p><i class="bi bi-person-circle"></i> Pilih Aktor</p>
        <p><i class="bi bi-arrow-down"></i></p>
        <p><i class="bi bi-cpu"></i> Analisis Filmografi</p>
        <p><i class="bi bi-arrow-down"></i></p>
        <p><i class="bi bi-graph-up"></i> Hitung Similarity</p>
        <p><i class="bi bi-arrow-down"></i></p>
        <p><i class="bi bi-film"></i> Dapatkan Rekomendasi</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Perbaikan: tidak menggunakan unsafe_allow_html untuk st.success
    st.markdown("""
    <div style="background-color: #d4edda; color: #155724; padding: 12px; border-radius: 8px; border: 1px solid #c3e6cb;">
    <i class="bi bi-arrow-right-circle"></i> <strong>Sukses!</strong> Coba buka menu <strong>Cari Rekomendasi Film</strong> untuk mulai eksplorasi!
    </div>
    """, unsafe_allow_html=True)

# ===============================
# HALAMAN 3 — REKOMENDASI FILM
# ===============================
elif selected_menu == "🔍 Cari Rekomendasi Film":
    st.markdown('<h1><i class="bi bi-search header-icon"></i> Cari Rekomendasi Film</h1>', unsafe_allow_html=True)
    st.markdown('<p>Temukan film serupa berdasarkan aktor favoritmu <i class="bi bi-person-circle"></i></p>', unsafe_allow_html=True)

    actor_list = sorted(df["Actor"].unique().tolist())
    
    # Perbaikan: menggunakan markdown untuk label dengan icon
    st.markdown('<p><i class="bi bi-search"></i> Pilih atau ketik nama aktor:</p>', unsafe_allow_html=True)
    selected_actor = st.selectbox(
        "Pilih aktor:",
        actor_list,
        index=None,
        placeholder="Ketik nama aktor...",
        label_visibility="collapsed"  # Sembunyikan label default
    )

    if selected_actor:
        # --- Filmografi ---
        actor_films = df[df["Actor"] == selected_actor]["Film"].tolist()
        st.markdown(f'### <i class="bi bi-collection-play"></i> Filmografi {selected_actor}', unsafe_allow_html=True)
        
        if actor_films:
            films_display = "\n".join([f"- <i class='bi bi-film'></i> {film}" for film in actor_films[:15]])
            st.markdown(f'<div class="section">{films_display}</div>', unsafe_allow_html=True)
            
            if len(actor_films) > 15:
                st.markdown(f"""
                <div style="background-color: #d1ecf1; color: #0c5460; padding: 10px; border-radius: 5px; border: 1px solid #bee5eb;">
                <i class="bi bi-info-circle"></i> Menampilkan 15 dari {len(actor_films)} film
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; border: 1px solid #ffeaa7;">
            <i class="bi bi-exclamation-triangle"></i> Tidak ada data film untuk aktor ini
            </div>
            """, unsafe_allow_html=True)

        # --- Similarity antar aktor ---
        st.markdown('### <i class="bi bi-cpu"></i> Menghitung Rekomendasi...', unsafe_allow_html=True)
        
        with st.spinner('Menganalisis filmografi dan mencari aktor serupa...'):
            actor_group = df.groupby("Actor")["Film"].apply(lambda x: " ".join(x)).reset_index()
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(actor_group["Film"])
            similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
            sim_df = pd.DataFrame(similarity, index=actor_group["Actor"], columns=actor_group["Actor"])

            similar_actors = sim_df[selected_actor].sort_values(ascending=False)[1:6]

        # --- Aktor Serupa ---
        st.markdown('### <i class="bi bi-people-fill"></i> Aktor dengan Filmografi Serupa', unsafe_allow_html=True)
        
        for i, (actor, score) in enumerate(similar_actors.items(), start=1):
            progress_value = min(score * 100, 100)
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <strong>{i}. {actor}</strong>
                <div style="background-color: #e9ecef; border-radius: 10px; height: 20px; margin: 5px 0;">
                    <div style="background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); width: {progress_value}%; height: 100%; border-radius: 10px; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; color: white; font-size: 0.8em;">
                        {score:.3f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Rekomendasi Film ---
        st.markdown('### <i class="bi bi-ticket-perforated"></i> Rekomendasi Film dari Aktor Serupa', unsafe_allow_html=True)
        
        similar_actors_list = similar_actors.index.tolist()
        recommended_films = (
            df[df["Actor"].isin(similar_actors_list)]
            .sort_values(by="Rating", ascending=False)
            .drop_duplicates(subset=["Film"])
            .head(10)
        )

        if not recommended_films.empty:
            # Format dataframe dengan styling
            styled_films = recommended_films[["Film", "Actor", "Year", "Rating", "Votes"]].copy()
            styled_films["Rating"] = styled_films["Rating"].round(2)
            
            # Tampilkan dengan container yang lebih menarik
            for _, film in styled_films.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.markdown(f"**<i class='bi bi-film'></i> {film['Film']}** ({film['Year']})", unsafe_allow_html=True)
                        st.markdown(f"<small><i class='bi bi-person'></i> {film['Actor']}</small>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<i class='bi bi-star-fill'></i> **{film['Rating']}**/10", unsafe_allow_html=True)
                        st.markdown(f"<small><i class='bi bi-eye'></i> {film['Votes']:,} votes</small>", unsafe_allow_html=True)
                    with col3:
                        if film['Rating'] >= 8.0:
                            st.markdown('<span style="color: green;"><i class="bi bi-award"></i> Excellent</span>', unsafe_allow_html=True)
                        elif film['Rating'] >= 7.0:
                            st.markdown('<span style="color: orange;"><i class="bi bi-hand-thumbs-up"></i> Good</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span style="color: blue;"><i class="bi bi-film"></i> Watch</span>', unsafe_allow_html=True)
                    st.divider()
        else:
            st.markdown("""
            <div style="background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; border: 1px solid #ffeaa7;">
            <i class="bi bi-exclamation-triangle"></i> Tidak ada rekomendasi film yang ditemukan
            </div>
            """, unsafe_allow_html=True)

    else:
        # Perbaikan: tidak menggunakan unsafe_allow_html untuk st.info
        st.markdown("""
        <div style="background-color: #d1ecf1; color: #0c5460; padding: 12px; border-radius: 8px; border: 1px solid #bee5eb;">
        <i class="bi bi-info-circle"></i> Silakan pilih aktor terlebih dahulu untuk melihat rekomendasi film.
        </div>
        """, unsafe_allow_html=True)

# ===============================
# HALAMAN 4 — DATASET & VISUALISASI
# ===============================
elif selected_menu == "📊 Dataset & Visualisasi":
    st.markdown('<h1><i class="bi bi-bar-chart-line header-icon"></i> Dataset dan Visualisasi</h1>', unsafe_allow_html=True)

    # Perbaikan: menggunakan emoji untuk tab karena Streamlit tidak mendukung HTML di tab
    tab1, tab2 = st.tabs(["🎬 Data Film", "👤 Data Aktor"])

    # --- TAB 1: Data Film ---
    with tab1:
        st.markdown('<h3><i class="bi bi-film"></i> Data Film</h3>', unsafe_allow_html=True)
        
        # Statistik singkat
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_films = df["Film"].nunique()
            st.metric("Total Film", f"{total_films:,}")
        with col2:
            avg_rating = df["Rating"].mean()
            st.metric("Rata-rata Rating", f"{avg_rating:.2f}")
        with col3:
            latest_year = df["Year"].max()
            st.metric("Tahun Terbaru", latest_year)
        with col4:
            top_rating = df["Rating"].max()
            st.metric("Rating Tertinggi", f"{top_rating:.1f}")

        film_data = df[["Film", "Year", "Rating", "Votes"]].drop_duplicates()
        st.dataframe(film_data, use_container_width=True)

        # Visualisasi
        col1, col2 = st.columns(2)
        
        with col1:
            # Grafik distribusi rating (berdasarkan film unik)
            fig = px.histogram(film_data, x="Rating", nbins=20, 
                             title="Distribusi Rating Film",
                             color_discrete_sequence=["#2b2d42"])
            fig.update_layout(
                showlegend=False,
                yaxis_title="Banyak Film",
                xaxis_title="Rating"
            )
            # Format angka tanpa "k" di belakang
            fig.update_layout(
                yaxis=dict(tickformat=".0f"),
                xaxis=dict(tickformat=".1f")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Grafik jumlah film per tahun
            yearly_counts = film_data.groupby("Year").size().reset_index(name="Jumlah Film")
            fig2 = px.bar(yearly_counts, x="Year", y="Jumlah Film", 
                         title="Jumlah Film per Tahun",
                         color_discrete_sequence=["#ef233c"])
            fig2.update_layout(
                yaxis_title="Banyak Film",
                xaxis_title="Tahun",
                yaxis=dict(tickformat=".0f")
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Film dengan rating tertinggi
        st.markdown('<h4><i class="bi bi-trophy"></i> Film dengan Rating Tertinggi</h4>', unsafe_allow_html=True)
        top_films = film_data.sort_values(by="Rating", ascending=False).head(10)
        fig3 = px.bar(top_films, x="Rating", y="Film", orientation="h",
                     title="Top 10 Film dengan Rating Tertinggi", 
                     color="Rating", color_continuous_scale="Viridis")
        st.plotly_chart(fig3, use_container_width=True)

        # Detail film
        st.markdown("---")
        st.markdown('<h4><i class="bi bi-zoom-in"></i> Lihat Detail Film</h4>', unsafe_allow_html=True)
        selected_film = st.selectbox("Pilih film untuk melihat aktor yang berperan:", sorted(df["Film"].unique()), key="film_select")
        if selected_film:
            film_detail = df[df["Film"] == selected_film][["Actor", "Rating"]].sort_values(by="Rating", ascending=False)
            st.markdown(f'### <i class="bi bi-film"></i> {selected_film}', unsafe_allow_html=True)
            st.dataframe(film_detail, use_container_width=True)

    # --- TAB 2: Data Aktor ---
    with tab2:
        st.markdown('<h3><i class="bi bi-person-badge"></i> Data Aktor</h3>', unsafe_allow_html=True)
        
        actor_summary = df.groupby("Actor").agg({
            "Film": "count",
            "Rating": "mean",
            "Votes": "sum"
        }).rename(columns={
            "Film": "Jumlah Film", 
            "Rating": "Rata-rata Rating", 
            "Votes": "Total Votes"
        }).reset_index()

        # Statistik aktor
        col1, col2, col3 = st.columns(3)
        with col1:
            total_actors = len(actor_summary)
            st.metric("Total Aktor", f"{total_actors:,}")
        with col2:
            avg_films_per_actor = actor_summary["Jumlah Film"].mean()
            st.metric("Rata-rata Film per Aktor", f"{avg_films_per_actor:.1f}")
        with col3:
            top_actor_rating = actor_summary["Rata-rata Rating"].max()
            st.metric("Rating Aktor Tertinggi", f"{top_actor_rating:.2f}")

        st.dataframe(actor_summary, use_container_width=True)

        # Visualisasi aktor
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 aktor dengan rating tertinggi
            top_actors = actor_summary.nlargest(10, "Rata-rata Rating")
            fig4 = px.bar(top_actors, x="Rata-rata Rating", y="Actor", orientation="h",
                         title="Top 10 Aktor dengan Rating Tertinggi",
                         color="Rata-rata Rating", color_continuous_scale="RdBu")
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            # Top 10 aktor dengan film terbanyak
            prolific_actors = actor_summary.nlargest(10, "Jumlah Film")
            fig5 = px.bar(prolific_actors, x="Jumlah Film", y="Actor", orientation="h",
                         title="Top 10 Aktor dengan Film Terbanyak",
                         color="Jumlah Film", color_continuous_scale="Blues")
            st.plotly_chart(fig5, use_container_width=True)

        # Detail aktor
        st.markdown("---")
        st.markdown('<h4><i class="bi bi-zoom-in"></i> Lihat Detail Aktor</h4>', unsafe_allow_html=True)
        selected_actor = st.selectbox("Pilih nama aktor untuk melihat film yang dibintanginya:", sorted(df["Actor"].unique()), key="actor_select")
        if selected_actor:
            actor_detail = df[df["Actor"] == selected_actor][["Film", "Year", "Rating"]].drop_duplicates()
            st.markdown(f'### <i class="bi bi-person-circle"></i> {selected_actor}', unsafe_allow_html=True)
            
            # Statistik aktor
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Film", len(actor_detail))
            with col2:
                st.metric("Rating Rata-rata", f"{actor_detail['Rating'].mean():.2f}")
            with col3:
                st.metric("Film Terbaru", int(actor_detail['Year'].max()))
            
            st.dataframe(actor_detail, use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<footer>
<hr>
<i class="bi bi-heart-fill" style="color: red;"></i> Dibuat dengan cinta oleh <b>Kelompok Kinder Joy</b> — Sistem Pemberi Rekomendasi 2025
<br><small><i class="bi bi-github"></i> Repository tersedia di GitHub</small>
</footer>
""", unsafe_allow_html=True)