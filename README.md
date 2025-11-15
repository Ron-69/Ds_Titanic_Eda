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
| **`Pclass`** | **Mediana = 3.0**, confirmando que a 3ª classe era a mais populosa. | Confirma ser uma variável altamente preditiva (status social). 

### 💡 Insights Chave da EDA Visual

A análise gráfica das relações entre as variáveis confirmou as hipóteses iniciais e orientou a Engenharia de Features:

1.  **Status Social e Gênero:** A sobrevivência foi fortemente influenciada pela `Pclass` e `Sex`.
    
    ![Taxa de Sobrevivência por Gênero e Classe de Bilhete](notebooks/plots/survival_rate_sex_pclass.png)

2.  **Idade e Outliers:** O Boxplot da Idade mostrou a distribuição em relação à sobrevivência.
    
    ![Distribuição da Idade por Sobrevivência](notebooks/plots/age_distribution_boxplot.png)

3.  **Tarifa (Fare):** A alta assimetria na tarifa foi confirmada visualmente, o que justificou a transformação logarítmica.
    
    ![Distribuição Bruta da Tarifa (Fare)](notebooks/plots/fare_distribution_histogram.png)

---|

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

## 3. Conclusões e Plano de Ação (Próximos Passos)

### 💡 Status da Preparação de Dados

A fase de preparação de dados foi finalizada, garantindo que o dataset esteja 100% numérico e pronto para o treinamento de modelos.

### 📊 Codificação e Seleção Final de Features

| Ação | Resultado | Dimensões Finais |
| :--- | :--- | :--- |
| **One-Hot Encoding (OHE)** | Aplicado em `Sex`, `Embarked`, `Title` e `Pclass`. | +8 Novas colunas binárias criadas. |
| **Seleção Final** | Colunas originais redundantes (`Name`, `Ticket`, `Cabin`, `SibSp`, `Parch`, `Fare` original) removidas. | DataFrame final com **15 colunas** (`Survived` + 14 Features). |
| **Divisão (Train/Test)** | Dados divididos em 80% Treino e 20% Teste. | Treino (`X_train`): **712 linhas** (80%). |

## 4. Resultados Finais e Conclusão

A fase final do projeto envolveu o treinamento de modelos, seguido pela otimização de hiperparâmetros e padronização dos dados para garantir o melhor desempenho geral.

### 4.1. Otimização e Validação do Random Forest via Grid Search

O modelo **Random Forest Classifier** foi submetido a um processo de **Otimização de Hiperparâmetros** utilizando **Grid Search com Validação Cruzada (CV=5)** para encontrar a combinação ideal de `n_estimators`, `max_depth`, `min_samples_split` e `min_samples_leaf`.

* **Melhores Parâmetros Encontrados (CV):** `max_depth=15`, `min_samples_leaf=2`, `n_estimators=100`.
* **Melhor Acurácia em CV:** **0.8330**.

A seguir, todos os modelos foram reavaliados utilizando **StandardScaler** nas *features* numéricas (`Age`, `Fare_Log`, `FamilySize`) para garantir que a escala não viesse a penalizar o desempenho.

### 🏆 4.2. Desempenho Final Consolidado dos Modelos

Apesar dos esforços de otimização, a **Regressão Logística Padronizada** manteve a melhor *performance* de generalização no conjunto de teste.

| Modelo | Acurácia (Teste) | Precision (Classe 1) | Recall (Classe 1) | Observação |
| :--- | :--- | :--- | :--- | :--- |
| **Regressão Logística (Baseline)** | 0.8156 | **0.79** | 0.71 | Melhor modelo inicial e vencedor da otimização. |
| **Regressão Logística (Padronizada)** | **0.8156** | 0.79 | 0.71 | **Não houve ganho**, indicando que as *features* criadas e o *log-transform* já haviam mitigado a sensibilidade de escala para este modelo. |
| **Random Forest (Otimizado/Padronizado)** | 0.8101 | 0.80 | 0.68 | Aumento da *Precision*, mas perda de *Recall*, não superando o *baseline*. |

### Conclusão Final do Projeto

1.  **Modelo Vencedor:** A **Regressão Logística** é o modelo final com **81.56% de acurácia**. Seu sucesso demonstra que as relações entre as *features* criadas (`Title`, `Has_Cabin`, etc.) e a sobrevivência são **predominantemente lineares**.
2.  **Fator de Sucesso:** A robustez da **Engenharia de Features** e do **Pré-processamento** (transformação logarítmica de `Fare` e imputação estratégica) foi o fator mais crucial para o desempenho.

---

## 🔗 Estrutura do Repositório

ds_titanic_eda_python/ ├── venv/ # Ignorado pelo Git (Ambiente Virtual) ├── notebooks/ │ ├── plots/ # Novo: Contém os gráficos da EDA Visual (.png) │ └── ds_titanic_eda.ipynb # Notebook principal com EDA, Feature Engineering e Modelagem ├── data/ │ └── Titanic-Dataset.csv # Dataset principal, usado para treino e análise. ├── README.md # Este arquivo (Documentação do Projeto) ├── requirements.txt # Lista de dependências └── .gitignore # Arquivo para exclusão de pastas (venv/) e arquivos de sistema