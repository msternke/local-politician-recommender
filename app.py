from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import app_tools

app = Flask(__name__)
app.vars = {}

#setting initial values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/local-race')
def local_race():
    return render_template('local-race.html')

@app.route('/features-coming-soon')
def features_coming_soon():
    return render_template('features-coming-soon.html')

@app.route('/local-race-analysis', methods = ['GET', 'POST'])
def local_race_analysis():
    if request.method == 'POST':
        #loading NYC mayor word matrix and creating its similarity matrix
        with open('data/word_matrices/nyc_mayor.pkl', 'rb') as file:
            nyc_mayor_word_matrix = pickle.load(file)
        cos_sim_df = app_tools.create_similarity_matrix(cosine_similarity, nyc_mayor_word_matrix)

        #run analysis for entered Twitter handle
        handle = request.form['handle']
        updated_word_matrix, updated_cos_sim_df = app_tools.determine_user_similarity(handle, nyc_mayor_word_matrix)
        similarity_vector = updated_cos_sim_df[handle].sort_values(ascending=False).iloc[1:].to_frame()

        #Making similiarty bar chart
        fig_similaity_bar = app_tools.make_sim_bar_chart(similarity_vector)


        #Running PCA on word matrix
        nyc_mayor_pca = PCA(n_components=2)
        nyc_mayor_pcs = nyc_mayor_pca.fit_transform(nyc_mayor_word_matrix)
        updated_pcs = nyc_mayor_pca.transform(updated_word_matrix)
        fig_pca = app_tools.make_pca_plot(nyc_mayor_pcs, [i.replace("-", " ") for i in nyc_mayor_word_matrix.index])
        fig_pca.circle(updated_pcs[-1,0], y = updated_pcs[-1,1], size=10,
                        color = 'red', legend_label = handle)

        script, div = components(fig_similaity_bar)

    else:
        return render_template('local-race.html')

    return render_template('local-race-analysis.html', script=script, div=div, entered_handle=handle)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
