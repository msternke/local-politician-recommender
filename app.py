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
        with open('data/nyc_mayor_info_all.csv', 'rb') as file:
            nyc_mayor_info = pd.read_csv(file)


        cos_sim_df = app_tools.create_similarity_matrix(cosine_similarity, nyc_mayor_word_matrix)

        #run analysis for entered Twitter handle
        handle = request.form['handle']
        most_least = request.form['most-least']
        updated_word_matrix, updated_cos_sim_df = app_tools.determine_user_similarity(handle, nyc_mayor_word_matrix)
        similarity_vector = updated_cos_sim_df[handle].sort_values(ascending=False).iloc[1:].to_frame()

        #Making similiarty bar chart
        fig_similaity_bar = app_tools.make_sim_bar_chart(similarity_vector)
        script, div = components(fig_similaity_bar)

        #chosen candidate info
        if most_least == 'Most':
            cand_index = 0
        else:
            cand_index = -1
        top_img = f"static/img/nyc_mayor_imgs/{similarity_vector.index[cand_index].replace(' ', '-')}.png"
        chosen_cand = similarity_vector.index[cand_index]

        cand_info = nyc_mayor_info[nyc_mayor_info['name'] == chosen_cand].iloc[0]
        website = cand_info['website']
        party = cand_info['party']
        if party == 'D':
            out_party = 'Democrat'
        elif party == 'R':
            out_party = 'Republican'
        else:
            out_party = 'Third party'

    else:
        return render_template('local-race.html')

    return render_template('local-race-analysis.html', script=script,
                            div=div, entered_handle=handle, top=chosen_cand,
                            top_img=top_img, most_least=most_least, website=website,
                            out_party=out_party)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
