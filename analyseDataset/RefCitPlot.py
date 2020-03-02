from plotly.offline import plot
import plotly.graph_objs as go
import json
from general.baseFileExtractor import get_seminal_s, get_survey_s, get_uninfluential_s, get_file_base

with open(get_survey_s(), encoding='latin-1') as s:
    survey_hlp = json.load(s)
with open(get_seminal_s(), encoding='latin-1') as s:
    seminal_hlp = json.load(s)
with open(get_uninfluential_s(), encoding='latin-1') as s:
    uninfluential_hlp = json.load(s)

title = 'Distribution of references and citations for publications of all three classes'

ref = []
cit = []

for p in survey_hlp['survey']:
    ref.append(len(p['ref']))
    cit.append(len(p['cit']))

trace1 = go.Scatter(
        mode='markers',
        x=cit,
        y=ref,
        marker=dict(
          color='orange',
          size=8,
          opacity=0.3,
          symbol='triangle-up',
          line=dict(
            color='orange',
            width=2
          )
        ),
        showlegend=False,
        name='Survey'
)

cit = []  # out
ref = []  # in

for p in seminal_hlp['seminal']:
    ref.append(len(p['ref']))
    cit.append(len(p['cit']))


trace2 = go.Scatter(
        mode='markers',
        x=cit,
        y=ref,
        marker=dict(
          color='blue',
          size=8,
          opacity=0.3,
          line=dict(
            color='blue',
            width=2
          )
        ),
        showlegend=False,
        name='Seminal'
)

cit = []  # out
ref = []  # in

for p in uninfluential_hlp['uninfluential']:
    ref.append(len(p['ref']))
    cit.append(len(p['cit']))


trace3 = go.Scatter(
        mode='markers',
        x=cit,
        y=ref,
        marker=dict(
          color='green',
          size=8,
          opacity=0.3, symbol='cross', line=dict(color='green', width=2)
        ),
        showlegend=False,
        name='Uninfluential'
)

data = [trace2, trace1, trace3]
layout = {'yaxis': dict(
              title='Number of references',
              type='log',
              autorange=True, gridcolor='#E2E2E2'
          ),
          'xaxis': dict(
              title='Number of citations',
              type='log',
              autorange=True, gridcolor='#E2E2E2'
          ), 'width': 1200, 'height': 500, 'paper_bgcolor': '#FFFFFF', 'plot_bgcolor': '#FFFFFF'
          }

fig = go.Figure(data=data, layout=layout)
plot(fig, get_file_base() + 'plots/' + title + '.pdf')
