import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando os dados
data = pd.read_csv('dados.csv', sep=';', encoding='latin1', dtype=str)

# Limpando dados (removendo valores nulos)
data_clean = data.dropna().copy()

# Convertendo colunas para valores numéricos
data_clean['NU_NOTA_CH'] = pd.to_numeric(data_clean['NU_NOTA_CH'], errors='coerce')
data_clean['TP_ESCOLA'] = pd.to_numeric(data_clean['TP_ESCOLA'], errors='coerce')
data_clean['TP_LOCALIZACAO_ESC'] = pd.to_numeric(data_clean['TP_LOCALIZACAO_ESC'], errors='coerce')

# Mapeamento de tipo de escola
escola_map = {
    1: 'Não respondeu',
    2: 'Pública',
    3: 'Privada'
}
data_clean['Tipo_Escola'] = data_clean['TP_ESCOLA'].map(escola_map)

# Mapeamento de localização da escola
localizacao_map = {
    1: 'Urbana',
    2: 'Rural'
}
data_clean['Localizacao_Escola'] = data_clean['TP_LOCALIZACAO_ESC'].map(localizacao_map)

# Tratamento de Outliers
Q1 = data_clean['NU_NOTA_CH'].quantile(0.25)
Q3 = data_clean['NU_NOTA_CH'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
data_clean = data_clean[(data_clean['NU_NOTA_CH'] >= lower_limit) & (data_clean['NU_NOTA_CH'] <= upper_limit)]

# ------------------- Estatísticas -------------------

# Estatísticas por Tipo de Escola
print("Estatísticas por Tipo de Escola:")
print(data_clean.groupby('Tipo_Escola')['NU_NOTA_CH'].describe())

# Estatísticas por Localização da Escola
print("\nEstatísticas por Localização da Escola:")
print(data_clean.groupby('Localizacao_Escola')['NU_NOTA_CH'].describe())

# Estatísticas por Estado
print("\nEstatísticas por Estado:")
print(data_clean.groupby('SG_UF_ESC')['NU_NOTA_CH'].describe())

# ------------------- Gráficos -------------------

# Histograma para Tipo de Escola
plt.figure(figsize=(12, 6))
sns.histplot(data=data_clean, x='NU_NOTA_CH', hue='Tipo_Escola', kde=True, palette='Set2', bins=30)
plt.title('Distribuição das Notas de Ciências Humanas por Tipo de Escola', fontsize=14)
plt.xlabel('Nota de Ciências Humanas', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.legend(labels=data_clean['Tipo_Escola'].unique(), title='Tipo de Escola', loc='upper right', fontsize=10)
plt.show()

# Histograma para Localização da Escola
plt.figure(figsize=(12, 6))
sns.histplot(data=data_clean, x='NU_NOTA_CH', hue='Localizacao_Escola', kde=True, palette='Set1', bins=30)
plt.title('Distribuição das Notas de Ciências Humanas por Localização da Escola', fontsize=14)
plt.xlabel('Nota de Ciências Humanas', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.legend(labels=data_clean['Localizacao_Escola'].unique(), title='Localização da Escola', loc='upper right', fontsize=10)
plt.show()

# Histograma para Estado
plt.figure(figsize=(14, 6))
sns.histplot(data=data_clean, x='NU_NOTA_CH', hue='SG_UF_ESC', kde=True, palette='tab20', bins=30, multiple='stack')
plt.title('Distribuição das Notas de Ciências Humanas por Estado', fontsize=14)
plt.xlabel('Nota de Ciências Humanas', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.legend(labels=data_clean['SG_UF_ESC'].unique(), title='Estado', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.xticks(rotation=45)
plt.show()
