import pandas as pd
import panel as pn
import hvplot.pandas
import holoviews as hv

# Configuration du thème et des extensions
pn.extension(design='material', sizing_mode='stretch_width')

# Chargement des données (version légère pour le web)
DATA_URL = 'creditcard_lite.csv'

def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        return pd.DataFrame({'Erreur': [str(e)], 'Class': [0], 'Amount': [0], 'V1': [0], 'V2': [0]})

df = load_data()

# --- Thème et Couleurs ---
ACCENT = "#0072B5"
FRAUD = "#D62728"
NORMAL = "#2CA02C"

# --- Composants du Dashboard ---

# 1. Métriques Clés
def get_metrics():
    total = len(df)
    frauds = len(df[df['Class'] == 1])
    rate = (frauds / total) * 100 if total > 0 else 0
    
    return pn.Row(
        pn.indicators.Number(name="Total Transactions", value=total, font_size='22pt', title_size='12pt'),
        pn.indicators.Number(name="Fraudes Identifiées", value=frauds, font_size='22pt', title_size='12pt', colors=[(1, FRAUD)]),
        pn.indicators.Number(name="Taux de Risque", value=rate, font_size='22pt', title_size='12pt', format='{value:.3f}%'),
        justify_content='space-around', margin=(10, 0)
    )

# 2. Visualisations
amount_dist = df.hvplot.hist(
    'Amount', bins=50, logy=True, height=400, color=NORMAL, alpha=0.7,
    title="Distribution des Montants (Log)", xlabel="Montant ($)", ylabel="Fréquence"
)

spatial_analysis = df.hvplot.scatter(
    x='V1', y='V2', c='Class', cmap=[NORMAL, FRAUD],
    title="Analyse de Densité V1/V2", height=400, alpha=0.5, size=40
)

# 3. Explorateur de Variables
var_list = ['Amount', 'V1', 'V2', 'V14', 'V17']
var_select = pn.widgets.Select(name='Analyser la variable :', options=var_list, value='V17')

@pn.depends(var_select)
def box_plot_view(var):
    return df.hvplot.box(
        y=var, by='Class', height=400, color=[NORMAL, FRAUD],
        title=f"Profil de la variable {var}", ylabel=var, legend=False
    )

# --- Layout Final ---
main_view = pn.Column(
    pn.pane.Markdown("# 🛡️ Dashboard de Surveillance Fraude"),
    get_metrics(),
    pn.Row(amount_dist, spatial_analysis),
    pn.Row(pn.Column("### Focus Variable", var_select, box_plot_view)),
    pn.pane.Markdown("### 🔍 Détails des Transactions Suspectes"),
    pn.widgets.DataFrame(df[df['Class'] == 1].head(15), sizing_mode='stretch_width', height=300),
    max_width=1100, align='center'
)

template = pn.template.MaterialTemplate(
    title="Détection de Fraude CB",
    header_background=ACCENT,
    main=[main_view]
)

template.servable()
