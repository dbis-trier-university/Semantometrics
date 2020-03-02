
# durchschnittsjahr für alle ref pro sem/sur berechnen
# Histograms mit Jahren von Cit/Ref von allen sem/sur Publikationen aus einem Jahr

from plotly.offline import plot
import plotly.graph_objs as go
import json
import numpy as np
from general.baseFileExtractor import get_seminal_s, get_survey_s, get_uninfluential_s, get_file_base

# read in
with open(get_survey_s(), encoding='latin-1') as s:
    survey_hlp = json.load(s)
with open(get_seminal_s(), encoding='latin-1') as s:
    seminal_hlp = json.load(s)
with open(get_uninfluential_s(), encoding='latin-1') as s:
    uninfluential_hlp = json.load(s)

sem_ref = []
sur_ref = []
uni_ref = []

for p in seminal_hlp['seminal']:
    for ref in p['ref']:
        sem_ref.append(ref['year'])

for p in survey_hlp['survey']:
    for ref in p['ref']:
        sur_ref.append(ref['year'])

for p in uninfluential_hlp['uninfluential']:
    for ref in p['ref']:
        uni_ref.append(ref['year'])

trace1 = go.Histogram(x=sem_ref, xbins=dict(start=1950, end=2019, size=1), marker=dict(color='blue'),
                      name='seminal references')

trace2 = go.Histogram(x=sur_ref, xbins=dict(start=1950, end=2019, size=1), marker=dict(color='orange'),
                      name='survey references')

trace3 = go.Histogram(x=uni_ref, xbins=dict(start=1950, end=2019, size=1), marker=dict(color='green'),
                      name='uninfluential references')


data = [trace1, trace2, trace3]
layout = go.Layout(showlegend=True, autosize=False, width=600, height=300,
                   margin=go.layout.Margin(l=50, r=15, b=40, t=10, pad=4), xaxis=dict(
                       title='Years', showgrid=False
                   ),
                   yaxis=dict(
                       title='Number of references', showgrid=True, gridcolor='#E2E2E2'
                   ),
                   legend=dict(
                       x=0.01,
                       y=1,
                       font=dict(
                           family='sans-serif',
                           size=12,
                           color='#000'
                       ),
                       bgcolor='#E2E2E2',
                       bordercolor='#FFFFFF',
                       borderwidth=2
                   ), paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF'
                   )
fig = go.Figure(data=data, layout=layout)

plot(fig, get_file_base() + 'plots/references', image='jpeg')

print(np.mean(sem_ref))
print(np.mean(sur_ref))
print(np.mean(uni_ref))
