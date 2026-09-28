import pandas as pd

def genre_stats(movies):
    rows=[]
    for _,r in movies.iterrows():
        for g in str(r.genres).split('|'):
            if g.strip(): rows.append((g.strip(), r.avg_rating, r.rating_count))
    return pd.DataFrame(rows, columns=['genre','avg_rating','rating_count'])

def platform_stats(movies):
    return movies.groupby('language').agg(movies=('movie_id','count'), avg_rating=('avg_rating','mean')).reset_index()
