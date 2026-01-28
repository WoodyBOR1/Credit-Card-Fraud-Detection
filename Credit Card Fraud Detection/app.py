import pandas as pd
import panel as pn
import hvplot.pandas
import holoviews as hv

pn.extension(design='material')

# Configuration for Pyodide: we use the lite version relative to the app
DATA_URL = 'creditcard_lite.csv'

def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        return pd.DataFrame({'Error': [f"Could not load data: {e}"]})

df = load_data()

# --- Components ---

title = pn.pane.Markdown("""
# Credit Card Fraud Detection Dashboard
### Exploratory Data Analysis & Pattern Identification
""", styles={'text-align': 'center'})

# Metrics
total_trans = df.shape[0]
fraud_count = df[df['Class'] == 1].shape[0]
fraud_rate = (fraud_count / total_trans) * 100

metrics = pn.Row(
    pn.indicators.Number(name="Total Transactions", value=total_trans, format='{value}', colors=[(100, 'black')], font_size='24pt'),
    pn.indicators.Number(name="Fraudulent", value=fraud_count, format='{value}', colors=[(1, 'red')], font_size='24pt'),
    pn.indicators.Number(name="Fraud Rate (%)", value=fraud_rate, format='{value:.2f}%', colors=[(5, 'orange')], font_size='24pt'),
    sizing_mode='stretch_width', justify_content='center'
)

# Plots
amount_hist = df.hvplot.hist(
    y='Amount', by='Class', bins=50, alpha=0.5, 
    title='Transaction Amount Distribution (Log Scale)', 
    logy=True, height=400, responsive=True, color=['#3498db', '#e74c3c']
)

scatter_v1_v2 = df.hvplot.scatter(
    x='V1', y='V2', c='Class', cmap=['#3498db', '#e74c3c'], 
    title='V1 vs V2 Clusters', alpha=0.6,
    height=400, responsive=True
)

# Variable Analysis Selector
var_select = pn.widgets.Select(name='Variable Select', options=['Amount', 'Time'] + [f'V{i}' for i in range(1, 29)])

@pn.depends(var_select)
def plot_variable(var):
    return df.hvplot.box(y=var, by='Class', title=f'{var} Distribution by Class', 
                         height=400, responsive=True, color=['#3498db', '#e74c3c'])

# Layout
sidebar = [
    pn.pane.Markdown("### Configuration"),
    var_select
]

main_content = pn.Column(
    metrics,
    pn.Row(amount_hist, scatter_v1_v2),
    pn.Row(plot_variable),
    pn.pane.Markdown("### Data Sample (Frauds)"),
    pn.widgets.DataFrame(df[df['Class'] == 1].head(20), sizing_mode='stretch_width')
)

template = pn.template.MaterialTemplate(
    title='Credit Card Fraud EDA',
    sidebar=sidebar,
    main=main_content
)

template.servable()
