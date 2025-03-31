import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Carregar os dados
data = pd.read_csv('dados.csv', sep=';', encoding='latin1', dtype=str)

data_clean = data.dropna().copy()
data_clean.loc[:, 'TP_ESCOLA'] = pd.to_numeric(data_clean['TP_ESCOLA'], errors='coerce')
data_clean.loc[:, 'NU_NOTA_CH'] = pd.to_numeric(data_clean['NU_NOTA_CH'], errors='coerce')

# Remover outliers
Q1 = data_clean['NU_NOTA_CH'].quantile(0.25)
Q3 = data_clean['NU_NOTA_CH'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
data_clean = data_clean[(data_clean['NU_NOTA_CH'] >= limite_inferior) & (data_clean['NU_NOTA_CH'] <= limite_superior)]

# Cálculo das estatísticas por município
data_municipio = data_clean.groupby('NO_MUNICIPIO_PROVA')['NU_NOTA_CH'].agg(['mean', 'std']).reset_index()
data_municipio = data_municipio.rename(columns={'mean': 'Media_CH', 'std': 'Desvio_CH'})

# Converter para numérico
data_municipio['Media_CH'] = pd.to_numeric(data_municipio['Media_CH'], errors='coerce')
data_municipio['Desvio_CH'] = pd.to_numeric(data_municipio['Desvio_CH'], errors='coerce')

# Municípios com maior e menor média
top_municipios = data_municipio.nlargest(10, 'Media_CH')
low_municipios = data_municipio.nsmallest(10, 'Media_CH')

print("Municípios com maiores médias:")
print(top_municipios[['NO_MUNICIPIO_PROVA', 'Media_CH']])

print("\nMunicípios com menores médias:")
print(low_municipios[['NO_MUNICIPIO_PROVA', 'Media_CH']])

# Municípios com maior e menor variação
maior_variacao = data_municipio.nlargest(10, 'Desvio_CH')
menor_variacao = data_municipio.nsmallest(10, 'Desvio_CH')

print("\nMunicípios com maior variação de notas:")
print(maior_variacao[['NO_MUNICIPIO_PROVA', 'Desvio_CH']])

print("\nMunicípios com menor variação de notas:")
print(menor_variacao[['NO_MUNICIPIO_PROVA', 'Desvio_CH']])

# Analisando relação com fatores socioeconômicos
escolaridade_pais_map = {
    'A': 'Nunca estudou',
    'B': 'Não completou a 4ª série/5º ano do Ensino Fundamental',
    'C': 'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental',
    'D': 'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio',
    'E': 'Completou o Ensino Médio, mas não completou a Faculdade',
    'F': 'Completou a Faculdade, mas não completou a Pós-graduação',
    'G': 'Completou a Pós-graduação',
    'H': 'Não sei'
}

data_clean['Escolaridade_Pais'] = data_clean['Q001'].map(escolaridade_pais_map)

data_soc = data_clean.groupby('NO_MUNICIPIO_PROVA').agg({
    'TP_ESCOLA': lambda x: (x == 3).mean()
}).reset_index()
data_soc = data_soc.rename(columns={'TP_ESCOLA': 'Proporcao_Escolas_Privadas'})

# Juntando com as médias
analise_final = data_municipio.merge(data_soc, on='NO_MUNICIPIO_PROVA')

# Correlações
corr_escolas, p_escolas = spearmanr(analise_final['Proporcao_Escolas_Privadas'], analise_final['Media_CH'])

print("\nCorrelação entre média das notas e proporção de escolas privadas:")
print(f"Correlação: {corr_escolas}, Valor-p: {p_escolas}")

# Histograma das médias por município
plt.figure(figsize=(12, 6))
sns.barplot(data=top_municipios, x='Media_CH', y='NO_MUNICIPIO_PROVA', palette="Blues_r")
plt.title("Municípios com Maiores Médias de Ciências Humanas")
plt.xlabel("Média da Nota de Ciências Humanas")
plt.ylabel("Município")
plt.show()

# Histograma da variação por município
plt.figure(figsize=(12, 6))
sns.barplot(data=maior_variacao, x='Desvio_CH', y='NO_MUNICIPIO_PROVA', palette="Reds_r")
plt.title("Municípios com Maior Variação das Notas de Ciências Humanas")
plt.xlabel("Desvio Padrão da Nota de Ciências Humanas")
plt.ylabel("Município")
plt.show()

# Gráfico de correlação entre proporção de escolas privadas e média das notas
plt.figure(figsize=(10, 5))
sns.scatterplot(x=analise_final['Proporcao_Escolas_Privadas'], y=analise_final['Media_CH'])
for i, row in analise_final.iterrows():
    plt.text(row['Proporcao_Escolas_Privadas'], row['Media_CH'], row['NO_MUNICIPIO_PROVA'], fontsize=8)
plt.title("Relação entre Proporção de Escolas Privadas e Notas de Ciências Humanas")
plt.xlabel("Proporção de Escolas Privadas")
plt.ylabel("Média da Nota de Ciências Humanas")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()