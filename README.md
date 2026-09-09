# Análise de Sentimentos IMDb

Pipeline reproduzível para analisar resenhas do IMDb, comparar modelos clássicos de classificação de sentimento e gerar um relatório HTML autocontido.

[Abrir o relatório publicado](https://adrianodantas-ai.github.io/proj_anatel/)

## O que o projeto faz

1. Lê e valida um CSV de resenhas com classes positiva e negativa.
2. Mede qualidade e características do corpus.
3. Limpa HTML, entidades, pontuação e stopwords, preservando `not`, `no`, `nor` e `never`.
4. Remove textos duplicados antes do split estratificado 70/15/15 de treino, validação e teste.
5. Compara Multinomial Naive Bayes e Regressão Logística sobre Bag of Words com unigramas e bigramas.
6. Seleciona pelo F1 de validação, faz validação cruzada de cinco folds no treino e apresenta métricas e erros no HTML.

O estudo também mede stemming e lematização apenas na validação. O conjunto de teste não participa de decisões de limpeza, vocabulário, escolha do modelo ou normalização.

## Requisitos e instalação

- Python 3.11 ou superior
- Dependências em `requirements.txt`

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Formato do CSV

O CSV precisa ter uma coluna de texto e uma de rótulo. Para o dataset IMDb usado no estudo, os valores são:

```csv
review,sentiment
"A thoughtful and engaging movie.",positive
"Not a good film.",negative
```

O arquivo de dados não é versionado. Por padrão, o comando procura `IMDB%20Dataset.csv` na raiz do repositório.

## Gerar o relatório

```powershell
.venv\Scripts\python src\build_report.py `
  --input "IMDB%20Dataset.csv" `
  --text-column review `
  --label-column sentiment `
  --positive-label positive `
  --negative-label negative `
  --output reports\imdb_analysis.html
```

Abra `reports\imdb_analysis.html` localmente ou acesse a [versão publicada](https://adrianodantas-ai.github.io/proj_anatel/). O HTML incorpora os agregados, métricas e amostras necessários para a análise; ele não incorpora o CSV bruto.

## Decisões técnicas

- Bag of Words é o baseline interpretável; `CountVectorizer` aprende o vocabulário exclusivamente no treino.
- As métricas incluem accuracy, precision, recall, F1 e matriz de confusão para cada modelo.
- Duplicatas com rótulos conflitantes interrompem a execução; duplicatas iguais são removidas antes do split para impedir vazamento.
- A semente do split e dos modelos é fixa em `42` para tornar os resultados reproduzíveis.
- O relatório é estático e publicado pelo GitHub Actions; apenas `reports/imdb_analysis.html` é enviado ao GitHub Pages como `index.html`.

## Testes

```powershell
.venv\Scripts\python -m pytest
```

Os testes cobrem validação do CSV, limpeza, deduplicação, isolamento entre conjuntos, vocabulário ajustado no treino, relatório HTML e o experimento de normalização.

## Estrutura

```text
src/build_report.py       pipeline e geração do relatório
src/text_normalization.py normalização por stemming e lematização
src/report_template.html  apresentação HTML autocontida
tests/test_pipeline.py    testes do pipeline
reports/imdb_analysis.html relatório publicado
.github/workflows/static.yml deploy no GitHub Pages
```

## Limitações

O escopo é deliberadamente um estudo supervisionado binário, sem API, banco de dados, treinamento no navegador, embeddings ou redes neurais. Resultados dependem do CSV fornecido e não devem ser interpretados como avaliação de sentimento fora do domínio das resenhas IMDb.
