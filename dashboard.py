import pandas as pd
import panel as pn
import hvplot.pandas
import holoviews as hv

pn.extension(design='material')

DATA_URL = 'creditcard_lite.csv'

def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

df = load_data()

# Styles
ACCENT_COLOR = "#0072B5"
FRAUD_COLOR = "#D62728"
NORMAL_COLOR = "#2CA02C"

# Components
total = len(df)
frauds = len(df[df['Class'] == 1])
rate = (frauds / total) * 100 if total > 0 else 0

metrics = pn.Row(
    pn.indicators.Number(name="Total Transactions", value=total, font_size='22pt'),
    pn.indicators.Number(name="Frauduleuses", value=frauds, font_size='22pt', colors=[(1, FRAUD_COLOR)]),
    pn.indicators.Number(name="Taux %", value=rate, font_size='22pt', format='{value:.3f}%'),
    sizing_mode='stretch_width', justify_content='space-around'
)

amount_plot = df.hvplot.hist(
    'Amount', bins=50, logy=True, title="Distribution des Montants (Log)",
    color=NORMAL_COLOR, height=350, responsive=True
)

scatter_plot = df.hvplot.scatter(
    x='V1', y='V2', c='Class', cmap=[NORMAL_COLOR, FRAUD_COLOR],
    title="Analyse V1 / V2", height=350, responsive=True
)

var_select = pn.widgets.Select(name='Variable', options=['Amount', 'V1', 'V2', 'V17', 'V14'], value='V17')

@pn.depends(var_select)
def get_box(var):
    return df.hvplot.box(y=var, by='Class', title=f"Profil {var}", height=350, responsive=True)

main_content = pn.Column(
    pn.pane.Markdown("# 🛡️ Détection de Fraude"),
    metrics,
    pn.Row(amount_plot, scatter_plot),
    pn.Row(var_select, get_box),
    pn.widgets.DataFrame(df[df['Class'] == 1].head(10), sizing_mode='stretch_width')
)

template = pn.template.MaterialTemplate(
    title="Dashboard Fraude",
    main=[main_content],
    header_background=ACCENT_COLOR
)

template.servable()
