# Imports
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from bokeh.io import curdoc, push_notebook, output_notebook
from bokeh.layouts import column, layout
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput, HoverTool
from bokeh.plotting import figure, show
from ipywidgets import interact, interactive, fixed, interact_manual

# Loading data
data = pd.read_csv('data/skincare_products_clean.csv')

# Preprocessing ingredients
data['clean_ingreds'] = data['clean_ingreds'].apply(lambda x: str(x).replace('[', '').replace(']', '').replace("'", '').replace('"', ''))
all_ingreds = []

for i in data['clean_ingreds']:
    ingreds_list = i.split(', ')
    all_ingreds.extend(ingreds_list)

all_ingreds = sorted(set(all_ingreds))
all_ingreds = [ingred.strip() for ingred in all_ingreds if ingred != '']

# Create one-hot encoding matrix
one_hot_list = [[0] * len(data) for _ in range(len(all_ingreds))]

for i, ingredient in enumerate(all_ingreds):
    data[ingredient] = data['clean_ingreds'].apply(lambda x: 1 if ingredient in x else 0)

# Visualizing similarities
svd = TruncatedSVD(n_components=150, n_iter=1000, random_state=6)
svd_features = svd.fit_transform(data[all_ingreds])
tsne = TSNE(n_components=2, n_iter=1000000, random_state=6)
tsne_features = tsne.fit_transform(svd_features)

data['X'] = tsne_features[:, 0]
data['Y'] = tsne_features[:, 1]

unique_types = ['Moisturiser', 'Serum', 'Oil', 'Mist', 'Balm', 'Mask', 'Peel',
                'Eye Care', 'Cleanser', 'Toner', 'Exfoliator', 'Bath Salts',
                'Body Wash', 'Bath Oil']

source = ColumnDataSource(data)

plot = figure(title="Mapped Similarities", width=800, height=600)
plot.xaxis.axis_label = "t-SNE 1"
plot.yaxis.axis_label = 't-SNE 2'

plot.circle(x='X', y='Y', source=source, fill_alpha=0.7, size=10, color='#c0a5e3', alpha=1)

plot.background_fill_color = "#E9E9E9"
plot.background_fill_alpha = 0.3

hover = HoverTool(tooltips=[('Product', '@product_name'), ('Price', '@price')])
plot.add_tools(hover)

def type_updater(product_type=unique_types[0]):
    new_data = {'X': data[data['product_type'] == product_type]['X'],
                'Y': data[data['product_type'] == product_type]['Y'],
                'product_name': data[data['product_type'] == product_type]['product_name'],
                'price': data[data['product_type'] == product_type]['price']}
    source.data = new_data
    push_notebook()

output_notebook()
show(plot, notebook_handle=True)

# Extracting brand names
brand_list = ["111skin", "a'kin", ... ]  # Complete the list with all brand names
brand_list = sorted(brand_list, key=len, reverse=True)

data['brand'] = data['product_name'].str.lower()
for i, brand in enumerate(brand_list):
    data['brand'] = data['brand'].str.replace(brand.lower(), brand.title())

data['brand'] = data['brand'].replace(['Aurelia Probiotic Skincare'], 'Aurelia Skincare')
data['brand'] = data['brand'].replace(['Avene'], 'Avène')
data['brand'] = data['brand'].replace(['Bloom And Blossom'], 'Bloom & Blossom')
data['brand'] = data['brand'].replace(['Dr Brandt'], 'Dr. Brandt')
data['brand'] = data['brand'].replace(['Dr Hauschka'], 'Dr. Hauschka')
data['brand'] = data['brand'].replace(["L'oreal Paris", 'L’oréal Paris'], "L'oréal Paris")

# Creating the recommendation function
def recommender(search):
    cs_list = []
    brands = []
    output = []
    binary_list = []
    idx = data[data['product_name'] == search].index.item()
    for i in all_ingreds:
        binary_list.append(data.iloc[idx][i])

    point1 = np.array(binary_list).reshape(1, -1)
    prod_type = data['product_type'][idx]
    brand_search = data['brand'][idx]
    data_by_type = data[data['product_type'] == prod_type]

    for j in range(data_by_type.shape[0]):
        binary_list2 = data_by_type.iloc[j][all_ingreds].tolist()
        point2 = np.array(binary_list2).reshape(1, -1)
        dot_product = np.dot(point1, point2.T)
        norm_1 = np.linalg.norm(point1)
        norm_2 = np.linalg.norm(point2)
        cos_sim = dot_product / (norm_1 * norm_2)
        cs_list.append(cos_sim.item())

    data_by_type['cos_sim'] = cs_list
    data_by_type = data_by_type.sort_values('cos_sim', ascending=False)
    data_by_type = data_by_type[data_by_type.product_name != search]

    for _, row in data_by_type.iterrows():
        brand = row['brand']
        if len(brands) == 0:
            if brand != brand_search:
                brands.append(brand)
                output.append(row[['product_name', 'cos_sim']])
        elif brands.count(brand) < 2:
            if brand != brand_search:
                brands.append(brand)
                output.append(row[['product_name', 'cos_sim']])

    print('\033[1m', 'Recommending products similar to', search, ':', '\033[0m')
    print(pd.DataFrame(output).head(5))

# Using function to get recommendations
recommender("Origins GinZing™ Energy-Boosting Tinted Moisturiser SPF40 50ml")
recommender('Avène Antirougeurs Jour Redness Relief Moisturizing Protecting Cream (40ml)')
recommender('Bondi Sands Everyday Liquid Gold Gradual Tanning Oil 270ml')
recommender('Sukin Rose Hip Oil (25ml)')
recommender('La Roche-Posay Anthelios Anti-Shine Sun Protection Invisible SPF50+ Face Mist 75ml')
recommender('Clinique Even Better Clinical Radical Dark Spot Corrector + Interrupter 30ml')
recommender("FOREO 'Serum Serum Serum' Micro-Capsule Youth Preserve")
recommender('Garnier Organic Argan Mist 150ml')
recommender('Shea Moisture 100% Virgin Coconut Oil Daily Hydration Body Wash 384ml')
recommender('JASON Soothing Aloe Vera Body Wash 887ml')
