import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------- Leitura dos Dados -------------------
data = pd.read_csv('dados.csv', sep=';', encoding='latin1', dtype=str)

data_clean = data.dropna().copy()

# Convertendo colunas para valores numéricos
data_clean.loc[:, 'NU_NOTA_CH'] = pd.to_numeric(data_clean['NU_NOTA_CH'], errors='coerce')
data_clean.loc[:, 'TP_FAIXA_ETARIA'] = pd.to_numeric(data_clean['TP_FAIXA_ETARIA'], errors='coerce')
data_clean.loc[:, 'TP_ESCOLA'] = pd.to_numeric(data_clean['TP_ESCOLA'], errors='coerce')

# Mapeamento de faixas etárias
faixa_etaria_map = {
    1: 'Menor que 17 anos',
    2: '17 anos',
    3: '18 anos',
    4: '19 anos',
    5: '20 anos',
    6: '21 anos',
    7: '22 anos',
    8: '23 anos',
    9: '24 anos',
    10: '25 anos',
    11: 'Entre 26 e 30 anos',
    12: 'Entre 31 e 35 anos',
    13: 'Entre 36 e 40 anos',
    14: 'Entre 41 e 45 anos',
    15: 'Entre 46 e 50 anos',
    16: 'Entre 51 e 55 anos',
    17: 'Entre 56 e 60 anos',
    18: 'Entre 61 e 65 anos',
    19: 'Entre 66 e 70 anos',
    20: 'Maior que 70 anos'
}

data_clean.loc[:, 'Faixa_Etaria'] = data_clean['TP_FAIXA_ETARIA'].map(faixa_etaria_map)

def categorizar_faixa_etaria(faixa):
    if faixa in ['Menor que 17 anos', '17 anos']:
        return 'Menor que 18 anos'
    elif faixa in ['18 anos', '19 anos', '20 anos', '21 anos', '22 anos', '23 anos', '24 anos', '25 anos']:
        return 'Entre 18 e 25 anos'
    else:
        return 'Maior que 25 anos'

data_clean.loc[:, 'Faixa_Etaria_Categorizada'] = data_clean['Faixa_Etaria'].apply(categorizar_faixa_etaria)

# Mapeamento de renda familiar
renda_map = {
    'A': 'Nenhuma renda',
    'B': 'Até 1.320',
    'C': 'De 1.320,01 até 1.980',
    'D': 'De 1.980,01 até 2.640',
    'E': 'De 2.640,01 até 3.300',
    'F': 'De 3.300,01 até 3.960',
    'G': 'De 3.960,01 até 5.280',
    'H': 'De 5.280,01 até 6.600',
    'I': 'De 6.600,01 até 7.920',
    'J': 'De 7.920,01 até 9.240',
    'K': 'De 9.240,01 até 10.560',
    'L': 'De 10.560,01 até 11.880',
    'M': 'De 11.880,01 até 13.200',
    'N': 'De 13.200,01 até 15.840',
    'O': 'De 15.840,01 até 19.800',
    'P': 'De 19.800,01 até 26.400',
    'Q': 'Acima de 26.400'
}

data_clean.loc[:, 'Renda_Familiar'] = data_clean['Q006'].map(renda_map)

data_clean['Renda_Familiar'] = data_clean['Renda_Familiar'].fillna('Desconhecido')

# Mapeamento de tipo de escola
escola_map = {
    1: 'Não respondeu',
    2: 'Pública',
    3: 'Privada'
}

data_clean.loc[:, 'Tipo_Escola'] = data_clean['TP_ESCOLA'].map(escola_map)

# ------------------- Tratamento de Outliers -------------------

# Identificando outliers para a coluna de notas de Ciências Humanas
Q1 = data_clean['NU_NOTA_CH'].quantile(0.25)
Q3 = data_clean['NU_NOTA_CH'].quantile(0.75)
IQR = Q3 - Q1

# Calculando limites inferior e superior
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print(f"Limite inferior: {limite_inferior}, Limite superior: {limite_superior}")

# Removendo valores atípicos fora dos limites
data_clean = data_clean[(data_clean['NU_NOTA_CH'] >= limite_inferior) & (data_clean['NU_NOTA_CH'] <= limite_superior)]

# ------------------- Estatísticas Detalhadas -------------------

# Estatísticas por faixa etária
print("\nEstatísticas por Faixa Etária Categorizada:")
print(data_clean.groupby('Faixa_Etaria_Categorizada')['NU_NOTA_CH'].describe())

# Estatísticas por tipo de escola
print("\nEstatísticas por Tipo de Escola:")
print(data_clean.groupby('Tipo_Escola')['NU_NOTA_CH'].describe())

# ------------------- Cálculo da Correlação -------------------

# Mapeando faixas de renda para valores numéricos
renda_ordinal = {
    'Nenhuma renda': 0,
    'Até 1.320': 1,
    'De 1.320,01 até 1.980': 2,
    'De 1.980,01 até 2.640': 3,
    'De 2.640,01 até 3.300': 4,
    'De 3.300,01 até 3.960': 5,
    'De 3.960,01 até 5.280': 6,
    'De 5.280,01 até 6.600': 7,
    'De 6.600,01 até 7.920': 8,
    'De 7.920,01 até 9.240': 9,
    'De 9.240,01 até 10.560': 10,
    'De 10.560,01 até 11.880': 11,
    'De 11.880,01 até 13.200': 12,
    'De 13.200,01 até 15.840': 13,
    'De 15.840,01 até 19.800': 14,
    'De 19.800,01 até 26.400': 15,
    'Acima de 26.400': 16
}
data_clean['Renda_Ordinal'] = data_clean['Renda_Familiar'].map(renda_ordinal)

# Calculando correlação
correlacao = data_clean[['Renda_Ordinal', 'NU_NOTA_CH']].corr()
print("Correlação entre Renda e Notas de Ciências Humanas:")
print(correlacao)

# ------------------- Gráficos -------------------

plt.figure(figsize=(10, 6))
sns.histplot(data_clean['NU_NOTA_CH'], kde=True, bins=30, color='green')
plt.title('Distribuição das Notas de Ciências Humanas (Após Tratamento de Outliers)')
plt.xlabel('Nota de Ciências Humanas')
plt.ylabel('Frequência')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=data_clean, x='Faixa_Etaria_Categorizada', palette='viridis')
plt.title('Distribuição das Faixas Etárias')
plt.xlabel('Faixa Etária')
plt.ylabel('Número de Participantes')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=data_clean, x='NU_NOTA_CH', hue='Faixa_Etaria_Categorizada', multiple="stack", palette="viridis", bins=30)
plt.title('Distribuição das Notas de Ciências Humanas por Faixa Etária')
plt.xlabel('Nota de Ciências Humanas')
plt.ylabel('Frequência')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(data=data_clean, x='NU_NOTA_CH', hue='Renda_Familiar', multiple="stack", palette="viridis", bins=30)
plt.title('Distribuição das Notas de Ciências Humanas por Renda Familiar')
plt.xlabel('Nota de Ciências Humanas')
plt.ylabel('Frequência')
plt.xticks(rotation=45)
plt.show()

# Plot do Tipo de Escola
plt.figure(figsize=(12, 6))
sns.histplot(data=data_clean, x='NU_NOTA_CH', hue='Tipo_Escola', multiple="stack", palette="viridis", bins=30)
plt.title('Distribuição das Notas de Ciências Humanas por Tipo de Escola')  # Título completo
plt.xlabel('Nota de Ciências Humanas')
plt.ylabel('Frequência')
plt.xticks(rotation=45)  # Rotaciona os rótulos no eixo X para melhor visualização
plt.show()
