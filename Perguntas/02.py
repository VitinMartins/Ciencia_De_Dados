import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

data = pd.read_csv('dados.csv', sep=';', encoding='latin1', dtype=str)

data_clean = data.dropna().copy()
data_clean.loc[:, 'TP_ESCOLA'] = pd.to_numeric(data_clean['TP_ESCOLA'], errors='coerce')
data_clean.loc[:, 'NU_NOTA_CH'] = pd.to_numeric(data_clean['NU_NOTA_CH'], errors='coerce')

# ------------------- Tratamento de Outliers -------------------
Q1 = data_clean['NU_NOTA_CH'].quantile(0.25)
Q3 = data_clean['NU_NOTA_CH'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print(f"Limite inferior: {limite_inferior}, Limite superior: {limite_superior}")

data_clean = data_clean[(data_clean['NU_NOTA_CH'] >= limite_inferior) & (data_clean['NU_NOTA_CH'] <= limite_superior)]

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

internet_access_map = {
    'A': 'Não',
    'B': 'Sim'
}

data_clean['Escolaridade_Pais'] = data_clean['Q001'].map(escolaridade_pais_map)
data_clean['Acesso_Internet'] = data_clean['Q025'].map(internet_access_map)

escolaridade_pais_map_inv = {
    'Nunca estudou': 0,
    'Não completou a 4ª série/5º ano do Ensino Fundamental': 1,
    'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental': 2,
    'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio': 3,
    'Completou o Ensino Médio, mas não completou a Faculdade': 4,
    'Completou a Faculdade, mas não completou a Pós-graduação': 5,
    'Completou a Pós-graduação': 6,
    'Não sei': 7
}

data_clean['Escolaridade_Pais_Num'] = data_clean['Escolaridade_Pais'].map(escolaridade_pais_map_inv)

# ------------------- Análises Estatísticas -------------------

stats_notas = data_clean['NU_NOTA_CH'].describe()
print("Estatísticas Gerais - Notas de Ciências Humanas")
print(stats_notas)

stats_internet = data_clean.groupby('Acesso_Internet')['NU_NOTA_CH'].describe()
print("\nEstatísticas por Acesso à Internet")
print(stats_internet)

stats_tipo_escola = data_clean.groupby('TP_ESCOLA')['NU_NOTA_CH'].describe()
print("\nEstatísticas por Tipo de Escola")
print(stats_tipo_escola)

stats_escolaridade = data_clean.groupby('Escolaridade_Pais')['NU_NOTA_CH'].describe()
print("\nEstatísticas por Escolaridade dos Pais")
print(stats_escolaridade)

correlation, p_value = spearmanr(data_clean['Escolaridade_Pais_Num'], data_clean['NU_NOTA_CH'])
print("\nCorrelação de Spearman - Escolaridade dos Pais e Notas de Ciências Humanas")
print(f"Correlação: {correlation}")
print(f"Valor-p: {p_value}")

# ------------------- Gráficos -------------------

plt.figure(figsize=(10, 5))
sns.histplot(data_clean['NU_NOTA_CH'], bins=30, kde=False, color="blue", alpha=0.7)
plt.title("Distribuição de Notas de Ciências Humanas")
plt.xlabel("Nota de Ciências Humanas")
plt.ylabel("Número de Participantes")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(12, 6))
ax = sns.histplot(data=data_clean, x='NU_NOTA_CH', hue='Escolaridade_Pais', multiple="stack", palette="Set2", bins=30)
plt.title('Distribuição das Notas de Ciências Humanas por Escolaridade dos Pais')
plt.xlabel('Nota de Ciências Humanas')
plt.ylabel('Número de Participantes')

handles, labels = ax.get_legend_handles_labels()
if handles:
    plt.legend(handles=handles, labels=labels, title="Escolaridade dos Pais", bbox_to_anchor=(1.05, 0.5), loc='center left')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(12, 6))
ax = sns.histplot(
    data=data_clean,
    x='NU_NOTA_CH',
    hue='TP_ESCOLA',
    multiple='stack',
    palette='Set2',
    bins=30
)
plt.title('Distribuição das Notas de Ciências Humanas por Tipo de Escola')
plt.xlabel('Nota de Ciências Humanas')
plt.ylabel('Número de Participantes')
plt.legend(title="Tipo de Escola", labels=["Não respondeu", "Pública", "Privada"], bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(12, 6))
ax = sns.histplot(
    data=data_clean,
    x='NU_NOTA_CH',
    hue='Acesso_Internet',
    multiple='stack',
    palette='Set2',
    bins=30
)
plt.title('Distribuição das Notas de Ciências Humanas por Acesso à Internet')
plt.xlabel('Nota de Ciências Humanas')
plt.ylabel('Número de Participantes')
plt.legend(title="Acesso à Internet", labels=["Não", "Sim"], bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
