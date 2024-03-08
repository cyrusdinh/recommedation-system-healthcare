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
for i in range(len(data['clean_ingreds'])):
    data['clean_ingreds'].iloc[i] = str(data['clean_ingreds'].iloc[i]).replace('[', '').replace(']', '').replace("'", '').replace('"', '')

all_ingreds = []
for i in data['clean_ingreds']:
    ingreds_list = i.split(', ')
    for j in ingreds_list:
        all_ingreds.append(j)

all_ingreds = sorted(set(all_ingreds))
all_ingreds.remove('')
for i in range(len(all_ingreds)):
    if all_ingreds[i][-1] == ' ':
        all_ingreds[i] = all_ingreds[i][0:-1]

one_hot_list = [[0] * 0 for i in range(len(all_ingreds))]

for i in data['clean_ingreds']:
    k = 0
    for j in all_ingreds:
        if j in i:
            one_hot_list[k].append(1)
        else:
            one_hot_list[k].append(0)
        k += 1

ingred_matrix = pd.DataFrame(one_hot_list).transpose()
ingred_matrix.columns = [sorted(set(all_ingreds))]

# Visualizing similarities
svd = TruncatedSVD(n_components=150, n_iter=1000, random_state=6)
svd_features = svd.fit_transform(ingred_matrix)
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

plot.circle(x='X', y='Y', source=source, fill_alpha=0.7, size=10,
            color='#c0a5e3', alpha=1)

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
k = 0
for i in data['brand']:
    for j in brand_list:
        if j in i:
            data['brand'][k] = data['brand'][k].replace(i, j.title())
    k += 1

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
    for i in ingred_matrix.iloc[idx][1:]:
        binary_list.append(i)
    point1 = np.array(binary_list).reshape(1, -1)
    point1 = [val for sublist in point1 for val in sublist]
    prod_type = data['product_type'][data['product_name'] == search].iat[0]
    brand_search = data['brand'][data['product_name'] == search].iat[0]
    data_by_type = data[data['product_type'] == prod_type]

    for j in range(data_by_type.index[0], data_by_type.index[0] + len(data_by_type)):
        binary_list2 = []
        for k in ingred_matrix.iloc[j][1:]:
            binary_list2.append(k)
        point2 = np.array(binary_list2).reshape(1, -1)
        point2 = [val for sublist in point2 for val in sublist]
        dot_product = np.dot(point1, point2)
        norm_1 = np.linalg.norm(point1)
        norm_2 = np.linalg.norm(point2)
        cos_sim = dot_product / (norm_1 * norm_2)
        cs_list.append(cos_sim)
    data_by_type = pd.DataFrame(data_by_type)
    data_by_type['cos_sim'] = cs_list
    data_by_type = data_by_type.sort_values('cos_sim', ascending=False)
    data_by_type = data_by_type[data_by_type.product_name != search]
    l = 0
    for m in range(len(data_by_type)):
        brand = data_by_type['brand'].iloc[l]
        if len(brands) == 0:
            if brand != brand_search:
                brands.append(brand)
                output.append(data_by_type.iloc[l])
        elif brands.count(brand) < 2:
            if brand != brand_search:
                brands.append(brand)
                output.append(data_by_type.iloc[l])
        l += 1

    return print('\033[1m', 'Recommending products similar to', search,':', '\033[0m'), print(pd.DataFrame(output)[['product_name', 'cos_sim']].head(5))

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
