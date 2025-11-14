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

## 2. Abordagem da Análise Exploratória de Dados (EDA)

A EDA foi conduzida com foco em traduzir características complexas em *features* preditivas, seguindo o seguinte fluxo:

### 2.1. Análise de Qualidade de Dados e Distribuição

* **Identificação de Nulos:** Utilização de `df.isnull().sum()` para quantificar a perda de dados em colunas críticas (`Age`, `Cabin`, `Embarked`).
* **Análise de Assimetria (Skewness):** Uso do `seaborn.histplot` e `scipy.stats.skew` na coluna `Fare` (Tarifa) para confirmar a **assimetria para a direita**. Este *insight* direciona a aplicação de uma **transformação logarítmica** no Pré-processamento para normalizar a distribuição e melhorar a performance de modelos.
* **Detecção de Outliers:** Utilização de `seaborn.boxplot` para visualizar a dispersão de `Age` e `Fare` e entender o impacto de valores extremos.

### 2.2. Engenharia de Features Chave (Feature Engineering)

* **Extração de Títulos:** A coluna `Name` foi explorada para extrair o **Título do Passageiro** (`Mr.`, `Mrs.`, `Master.`, `Rev.`, etc.). Este novo recurso é altamente preditivo, pois reflete o status social e a idade (ex: `Master` é usado para meninos, indicando uma alta probabilidade de serem salvos).
* **Engenharia Familiar:** As colunas `SibSp` (irmãos/cônjuges) e `Parch` (pais/filhos) foram combinadas para criar a *feature* **`FamilySize`**. Adicionalmente, foi criada a *feature* **`IsAlone`** (Se o passageiro viajava sozinho), um preditor conhecido por sua relevância na chance de sobrevivência.

---

## 3. Conclusões e Próximos Passos

### 💡 Insights Chave

1.  **Status Social:** A taxa de sobrevivência é diretamente proporcional à **Classe do Bilhete** (`Pclass`). Passageiros da 1ª Classe tiveram a maior probabilidade de sobrevivência, um *insight* confirmado pelo `seaborn.barplot`.
2.  **Idade e Gênero:** A regra "Mulheres e Crianças primeiro" é visível nos dados. A análise da idade versus sobrevivência (`seaborn.boxplot`) mostra uma clara vantagem para mulheres e crianças.
3.  **Tarifa (Fare):** A alta assimetria da tarifa e sua correlação com a `Pclass` reforçam que o poder de compra e o status eram os preditores mais fortes.

### 🚀 Próximos Passos

Com a EDA concluída e os *insights* de Engenharia de Features definidos, o projeto avança para a fase de **Modelagem**:

1.  **Tratamento de Nulos:** Imputação de `Age` (Mediana) e `Embarked` (Moda).
2.  **Codificação:** Aplicação de *One-Hot Encoding* nas variáveis categóricas relevantes.
3.  **Modelagem Preditiva:** Treinamento de modelos de Classificação (Regressão Logística, Random Forest) para prever `Survived`.

---

## 🔗 Estrutura do Repositório