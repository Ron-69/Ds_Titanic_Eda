# 🚢 Projeto de Data Science: EDA Avançada e Pré-processamento - Dataset Titanic

## 1. Visão Geral do Projeto

Este projeto demonstra um pipeline de **Análise Exploratória de Dados (EDA)** avançada e **Pré-processamento** para o clássico *dataset* Titanic. O objetivo principal é transformar dados brutos e complexos em um formato pronto para modelagem preditiva, focando na superação de desafios comuns em Data Science, como valores ausentes (`NaN`), *outliers* e variáveis categóricas de alta cardinalidade.

### 🎯 Objetivo Principal

Prever a **Sobrevivência** (`Survived` - Variável Target Binária) dos passageiros, construindo *features* robustas baseadas em *insights* estatísticos e de negócio.

### 🛠️ Tecnologias e Ferramentas

| Categoria | Ferramenta | Uso no Projeto |
| :--- | :--- | :--- |
| **Linguagem** | **Python** | Linguagem de programação principal. |
| **Ambiente** | **VS Code & Jupyter Notebooks** | Fluxo de trabalho profissional e reprodutível. |
| **Manipulação** | **Pandas & NumPy** | Carregamento, limpeza e transformação de dados. |
| **Visualização** | **Seaborn & Matplotlib** | EDA para identificar distribuições, *outliers* e relações. |
| **Estatística** | **SciPy** | Testes estatísticos rápidos (assimetria, testes t, etc.). |
| **Controle de Versão**| **Git** | Rastreamento de alterações e colaboração (branch `main`). |

---

## 2. Análise Exploratória de Dados (EDA)

A EDA foi conduzida com foco em traduzir características complexas em *features* preditivas, utilizando o rigor estatístico.

### 2.1. Análise de Qualidade de Dados e Estatísticas Sumárias

A primeira etapa envolveu o uso de **`df.info()`** e **`df.describe(include='all')`** para quantificar a qualidade e a distribuição inicial dos dados.

| Coluna | Descoberta Estatística Chave | Implicações para o Pré-processamento |
| :--- | :--- | :--- |
| **`Survived`**| Taxa de sobrevivência geral de **38.4%** (`Média = 0.3838`). | Indica um desbalanceamento moderado de classes. |
| **`Age`** | **20% de valores ausentes** (714 de 891). Média (29.7) e Mediana (28.0) próximas. | Será imputada com a **Mediana**, pois é mais robusta a *outliers*. |
| **`Fare`** | **Forte assimetria à direita** (Média $32.20 vs. Mediana $14.45). Max é $512. | **Transformação logarítmica** será obrigatória para mitigar a assimetria e o impacto dos *outliers*. |
| **`Cabin`** | **77% de valores ausentes** (204 de 891). | A coluna bruta será transformada em uma *feature* **binária** (`Has_Cabin`). |
| **`Embarked`** | Apenas **2 valores ausentes**. | Imputação simples pela **Moda** (Porto mais frequente). |
| **`Pclass`** | **Mediana = 3.0**, confirmando que a 3ª classe era a mais populosa. | Confirma ser uma variável altamente preditiva (status social). |

### 2.2. Engenharia de Features Chave (Feature Engineering)

Após a imputação de nulos (`Age` com Mediana, `Embarked` com Moda) e a transformação logarítmica de `Fare` (corrigindo a assimetria), as seguintes *features* preditivas foram criadas, gerando *insights* estatísticos robustos:

#### 💡 Resultados das Features Criadas

| Feature | Descrição | Taxa de Sobrevivência (Média) | Insight Chave |
| :--- | :--- | :--- | :--- |
| **`Has_Cabin` (1)**| Passageiro com cabine registrada | **66.67%** | Confirma que a posse de cabine é um poderoso preditor de status e sobrevivência (Taxa 2x maior que quem não tinha). |
| **`IsAlone` (0)** | Passageiro em grupo/família | **50.57%** | Passageiros que viajavam em grupo tiveram chance de sobrevivência significativamente maior do que os que viajavam sozinhos (30.35%). |
| **`Title` (Mrs)** | Título de Casada | **79.37%** | O `Title` provou ser o preditor mais forte, com `Mrs` e `Miss` apresentando as taxas mais altas. `Mr` (homem adulto) possui a taxa mais baixa (15.67%). |

---

## 3. Conclusões e Plano de Ação (Próximos Passos)

### 💡 Status das Fases

* ✅ **Imputação de Dados:** `Age` e `Embarked` foram tratados com sucesso.
* ✅ **Transformação de Dados:** `Fare` foi transformada via `log1p` para normalização.
* ✅ **Engenharia de Features:** `Has_Cabin`, `IsAlone`, `FamilySize` e `Title` foram criadas.

### 🚀 Próximos Passos no Pipeline

O projeto avança para a fase final de preparação de dados antes da modelagem:

1.  **Codificação:** Aplicação de **One-Hot Encoding** nas variáveis categóricas relevantes (`Sex`, `Embarked`, `Title`, `Pclass`).
2.  **Seleção Final:** Remoção de colunas originais que não serão mais usadas (`Name`, `Ticket`, `Cabin`, `Fare`, `SibSp`, `Parch`).
3.  **Modelagem Preditiva:** Treinamento e avaliação de modelos de Classificação (Regressão Logística, Random Forest) para prever `Survived`.

---

### 💡 Insights Chave da EDA Visual

Os gráficos de `seaborn.barplot` e `seaborn.boxplot` confirmaram:

1.  **Status Social:** A taxa de sobrevivência é diretamente proporcional à **Classe do Bilhete** (`Pclass`).
2.  **Idade e Gênero:** A regra "Mulheres e Crianças primeiro" é visível, sendo o **Gênero** o preditor categórico mais forte.
3.  **Tarifa (Fare):** A alta assimetria e sua correlação com a `Pclass` reforçam que o poder de compra era um fator determinante.

### 🚀 Próximos Passos no Pipeline

1.  **Pré-processamento:** Executar a imputação de nulos (`Age`, `Embarked`) e a transformação logarítmica de `Fare`.
2.  **Feature Engineering:** Implementar a criação de `Title`, `FamilySize`, `IsAlone` e `Has_Cabin`.
3.  **Codificação:** Aplicar **Codificação One-Hot** nas variáveis categóricas relevantes (ex: `Pclass`, `Embarked`, `Title`).
4.  **Modelagem Preditiva:** Treinamento e avaliação de modelos de Classificação (Regressão Logística, Random Forest).

---

## 🔗 Estrutura do Repositório

ds_titanic_eda_python/ ├── venv/ # Ignorado pelo Git (Ambiente Virtual) ├── notebooks/ │ └── ds_titanic_eda.ipynb # Notebook principal com EDA e código de Feature Engineering ├── data/ │ └── Titanic-Dataset.csv ├── README.md # Este arquivo ├── requirements.txt # Lista de dependências └── .gitignore # Arquivo para exclusão de pastas (venv/) e arquivos de sistema