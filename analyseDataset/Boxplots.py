from plotly.offline import plot
import plotly.graph_objs as go
from classify.Classification import read_in_csv_data_sem_sur_uni
from general.baseFileExtractor import get_file_base

vec = 'tfidf'
measure = 'cos'
stem = 'stemmed'

full_data, labels, sem, sur, uni = read_in_csv_data_sem_sur_uni(get_file_base() +
                                                                'extracted_features/tfidf_cos_stemmed.csv')
feature = 'sum'
group = 'A'
title = feature + group + ' ' + vec + ' ' + measure + ' ' + stem[:1]

sem = sem[feature + group]
sur = sur[feature + group]

uni = uni[feature + group]

trace1 = go.Box(x=sem, opacity=1, name='seminal', marker=dict(color='blue'))
trace2 = go.Box(x=sur, opacity=1, name='survey', marker=dict(color='orange'))
trace3 = go.Box(x=uni, opacity=1, name='uninfluential', marker=dict(color='green'))

layout = go.Layout(showlegend=False, autosize=False, width=800, height=250, xaxis_type='log',
                   margin=go.layout.Margin(l=50, r=15, b=40, t=10, pad=4),
                   xaxis=dict(
                       title='Value for ' + feature + group, showgrid=True, gridcolor='#E2E2E2'
                   ),
                   yaxis=dict(
                       showgrid=False
                   ),
                   paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF'
                   )

fig = go.Figure(layout=layout)
fig.add_trace(trace3)
fig.add_trace(trace2)
fig.add_trace(trace1)

plot(fig, get_file_base() + 'plots/' + title, image='jpeg')
