from plotly.offline import plot
import plotly.graph_objs as go
import json
from general.baseFileExtractor import get_seminal_s, get_survey_s, get_uninfluential_s, get_file_base

# read in
with open(get_survey_s(), encoding='latin-1') as s:
    survey_hlp = json.load(s)
with open(get_seminal_s(), encoding='latin-1') as s:
    seminal_hlp = json.load(s)
with open(get_uninfluential_s(), encoding='latin-1') as s:
    uninfluential_hlp = json.load(s)
seminal = 0  # 0 = seminal, 1 = survey, 2 = uninfluential

sem_p = []
sur_p = []
uni_p = []

sem_ref = []
sur_ref = []
uni_ref = []

sem_cit = []
sur_cit = []
uni_cit = []

for p in seminal_hlp['seminal']:
    sem_p.append(p['year'])

    for ref in p['ref']:
        sem_cit.append(ref['year'])

    for cit in p['cit']:
        sem_ref.append(cit['year'])

for p in survey_hlp['survey']:
    sur_p.append(p['year'])

    for ref in p['ref']:
        sur_cit.append(ref['year'])

    for cit in p['cit']:
        sur_ref.append(cit['year'])

for p in uninfluential_hlp['uninfluential']:
    uni_p.append(p['year'])

    for ref in p['ref']:
        uni_cit.append(ref['year'])

    for cit in p['cit']:
        uni_ref.append(cit['year'])

if seminal == 0:
    trace1 = go.Histogram(x=sem_p, opacity=1)
    trace2 = go.Histogram(x=sem_cit, opacity=0.5)
    trace3 = go.Histogram(x=sem_ref, opacity=0.5)
if seminal == 1:
    trace1 = go.Histogram(x=sur_p, opacity=1)
    trace2 = go.Histogram(x=sur_cit, opacity=0.5)
    trace3 = go.Histogram(x=sur_ref, opacity=0.5)
if seminal == 2:
    trace1 = go.Histogram(x=uni_p, opacity=1)
    trace2 = go.Histogram(x=uni_cit, opacity=0.5)
    trace3 = go.Histogram(x=uni_ref, opacity=0.5)

data = [trace1, trace2, trace3]
layout = go.Layout(showlegend=False, barmode='overlay', width=600, height=300,
                   margin=go.layout.Margin(l=50, r=15, b=40, t=10, pad=4),
                   yaxis=dict(range=[0, 6999]))
fig = go.Figure(data=data, layout=layout)

plot(fig, get_file_base() + 'plots/' + str(seminal), image='jpeg')
