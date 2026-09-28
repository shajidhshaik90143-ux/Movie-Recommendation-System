from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from src.config import MODEL_PATH
from src.data_loader import load_movies, load_ratings, prepare_movies
from src.recommender import MovieRecommender
from src.analytics import genre_stats

st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def get_data():
    raw_movies=load_movies(); ratings=load_ratings(); return prepare_movies(raw_movies,ratings), ratings

@st.cache_resource
def get_model():
    movies, _ = get_data()
    if MODEL_PATH.exists(): return MovieRecommender.load()
    model=MovieRecommender.train(movies); model.save(); return model

movies, ratings = get_data(); model=get_model()

st.markdown("""<style>
.hero{padding:2.2rem 2.5rem;border-radius:24px;background:linear-gradient(135deg,#17123b,#24145c 50%,#0d3b66);margin-bottom:1.5rem}
.hero h1{font-size:3rem;margin:0;color:#fff}.hero p{font-size:1.1rem;color:#dbeafe;max-width:800px}
.card{padding:1rem 1.1rem;border-radius:16px;background:#141b2d;border:1px solid #27324b;min-height:190px}
.badge{display:inline-block;padding:.2rem .55rem;border-radius:999px;background:#7c3aed;color:#fff;font-size:.75rem;margin:.15rem}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🎬 CineMatch AI</h1><p>Hybrid movie recommendations powered by content similarity, popularity signals, and personalized taste profiles.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🎯 Recommendation Controls")
    mode=st.radio("Mode",["Discover","Movie Similarity","Personalized","Analytics"])
    n=st.slider("Number of recommendations",4,15,8)
    st.divider()
    st.caption(f"Movies: {len(movies)}")
    st.caption(f"Ratings: {len(ratings):,}")
    st.caption(f"Users: {ratings.user_id.nunique()}")

if mode=="Discover":
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Movies",len(movies)); c2.metric("Users",ratings.user_id.nunique()); c3.metric("Ratings",f"{len(ratings):,}"); c4.metric("Avg rating",f"{ratings.rating.mean():.2f}/5")
    st.subheader("🔥 Popular right now")
    popular=movies.sort_values(["popularity_score","rating_count"],ascending=False).head(n)
    cols=st.columns(4)
    for i,(_,r) in enumerate(popular.iterrows()):
        with cols[i%4]:
            st.markdown(f'<div class="card"><h3>{r.title}</h3><span class="badge">{r.year}</span><span class="badge">⭐ {r.avg_rating:.1f}</span><p>{str(r.overview)[:120]}...</p><small>{r.genres.replace("|"," · ")}</small></div>',unsafe_allow_html=True)
    st.subheader("🔎 Explore the catalog")
    q=st.text_input("Search by title, cast, director, or keywords")
    genres=sorted({g.strip() for x in movies.genres for g in x.split('|') if g.strip()})
    selected=st.multiselect("Genres",genres)
    filtered=movies.copy()
    if q:
        mask=filtered.apply(lambda r:q.lower() in ' '.join(map(str,r.tolist())).lower(),axis=1); filtered=filtered[mask]
    if selected:
        filtered=filtered[filtered.genres.apply(lambda x:any(g in x.split('|') for g in selected))]
    st.dataframe(filtered[["title","year","genres","language","avg_rating","rating_count"]].sort_values("avg_rating",ascending=False),use_container_width=True,hide_index=True)

elif mode=="Movie Similarity":
    title=st.selectbox("Choose a movie",movies.title.tolist())
    source=movies[movies.title==title].iloc[0]
    st.markdown(f"### {source.title} ({source.year})")
    st.write(source.overview)
    st.caption(f"Genres: {source.genres.replace('|',' · ')}  |  Director: {source.director}  |  Rating: ⭐ {source.avg_rating:.1f}")
    recs=model.similar(source.movie_id,n=n)
    st.subheader("Movies you may also like")
    for _,r in recs.iterrows():
        a,b,c=st.columns([3,5,2])
        a.markdown(f"**{r.title}** ({r.year})")
        b.write(f"{r.genres.replace('|',' · ')} — {r.overview[:120]}...")
        c.write(f"⭐ {r.avg_rating:.1f}")
        st.caption("Why: "+model.explain(source.movie_id,r.movie_id))

elif mode=="Personalized":
    user=int(st.selectbox("Select demo user",sorted(ratings.user_id.unique())))
    history=ratings[ratings.user_id==user].merge(movies[["movie_id","title"]],on="movie_id").sort_values("rating",ascending=False)
    left,right=st.columns([1,2])
    with left:
        st.subheader("Your ratings")
        st.dataframe(history[["title","rating"]],hide_index=True,use_container_width=True)
    with right:
        recs=model.personalized(ratings,user,n=n)
        st.subheader("Personalized picks")
        if recs.empty: st.info("Not enough highly-rated movies for a profile yet.")
        else:
            st.dataframe(recs[["title","year","genres","avg_rating","personal_score"]],hide_index=True,use_container_width=True)

else:
    st.subheader("📊 Recommendation System Analytics")
    gs=genre_stats(movies).groupby('genre').agg(movies=('genre','size'),avg_rating=('avg_rating','mean'),ratings=('rating_count','sum')).reset_index().sort_values('movies',ascending=False)
    a,b=st.columns(2)
    with a: st.plotly_chart(px.bar(gs.head(12),x='genre',y='movies',title='Movies by genre'),use_container_width=True)
    with b: st.plotly_chart(px.bar(gs.head(12),x='genre',y='avg_rating',title='Average rating by genre'),use_container_width=True)
    st.plotly_chart(px.scatter(movies,x='rating_count',y='avg_rating',size='popularity_score',hover_name='title',title='Rating volume vs average rating'),use_container_width=True)

st.divider()
st.caption("CineMatch AI • Offline demo dataset • Replace data/movies.csv and data/ratings.csv with a properly licensed dataset for production use.")
