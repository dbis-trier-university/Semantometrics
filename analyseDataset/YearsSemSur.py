
# publication years for all seminal and survey papers

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

sem_p = []
sur_p = []
uni_p = []

for p in seminal_hlp['seminal']:
    sem_p.append(p['year'])

for p in survey_hlp['survey']:
    sur_p.append(p['year'])
for p in uninfluential_hlp['uninfluential']:
    uni_p.append(p['year'])

trace1 = go.Histogram(x=sem_p, xbins=dict(start=1969, end=2019, size=1), marker=dict(color='blue'), name='seminal')
trace2 = go.Histogram(x=sur_p, xbins=dict(start=1969, end=2019, size=1), marker=dict(color='orange'), name='survey')
trace3 = go.Histogram(x=uni_p, xbins=dict(start=1969, end=2019, size=1), marker=dict(color='green'),
                      name='uninfluential')

data = [trace1, trace2, trace3]
layout = go.Layout(showlegend=True, autosize=False, width=800, height=300,
                   margin=go.layout.Margin(l=50, r=15, b=40, t=10, pad=4),
                   xaxis=dict(
                       title='Years', showgrid=False
                   ),
                   yaxis=dict(
                       title='Number of publications', showgrid=True, gridcolor='#E2E2E2'
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

plot(fig, get_file_base() + 'plots/sem', image='jpeg')

print('mean year sem : ' + str(np.mean(sem_p)))
print('mean year sur : ' + str(np.mean(sur_p)))
print('mean year uni : ' + str(np.mean(uni_p)))
