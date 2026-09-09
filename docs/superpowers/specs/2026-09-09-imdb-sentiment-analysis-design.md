# Análise de Sentimentos IMDb — Design

**Data:** 2026-09-09  
**Status:** aprovado para planejamento; implementação ainda não iniciada

## Objetivo

Construir um projeto pessoal de análise de sentimentos que atenda ao formulário da entrevista técnica, demonstre cuidados contra vazamento e overfitting e termine em um relatório HTML estático, portátil e visualmente marcante.

O mesmo HTML deve permitir que uma pessoa selecione outro CSV local, indique explicitamente a coluna de texto, a coluna de classe e os valores correspondentes às classes positiva e negativa, e execute uma EDA básica no navegador. O arquivo selecionado não deve sair do computador da pessoa.

## Escopo

### Incluído

- EDA e verificações de qualidade do dataset IMDb.
- Limpeza de texto explicada e reproduzível.
- Split estratificado em treino, validação e teste sem resenhas duplicadas entre os conjuntos.
- Bag of Words com unigramas e bigramas.
- Multinomial Naive Bayes como baseline.
- Regressão Logística como segundo modelo.
- Accuracy, precision, recall, F1 e matriz de confusão.
- Validação cruzada do modelo escolhido pela validação.
- Análise curta de falsos positivos e falsos negativos.
- Relatório HTML autocontido com decisões, resultados, conclusões e próximos passos.
- EDA no navegador para CSVs futuros, sem treinamento de modelos.

### Não incluído

- Hospedagem ou publicação do relatório.
- Treinamento de modelos dentro do navegador.
- API, banco de dados, autenticação ou backend.
- Word embeddings, redes neurais ou ajuste extenso de hiperparâmetros.
- Stemming ou lematização.

## Dados e validação de entrada

O pipeline receberá o caminho do CSV e os nomes das colunas de texto e classe. Para o dataset atual, os valores esperados são `review`, `sentiment`, `positive` e `negative`.

Antes da análise, o pipeline deve validar:

- existência e leitura do arquivo;
- presença das colunas indicadas;
- textos e classes ausentes;
- classes inesperadas;
- quantidade de exemplos por classe;
- linhas malformadas;
- textos exatamente duplicados e eventuais conflitos de rótulo.

Erros de esquema, classes inválidas e duplicatas com rótulos conflitantes devem interromper a modelagem com uma mensagem clara. Problemas não impeditivos devem aparecer no relatório.

O CSV bruto será preservado e não será versionado no Git. A execução não produzirá uma cópia limpa permanente do dataset.

## Fluxo de dados

```text
IMDb CSV
  -> validação e EDA
  -> limpeza de texto
  -> remoção de duplicatas
  -> split estratificado 70/15/15
  -> Bag of Words ajustado somente no treino
  -> Naive Bayes + Regressão Logística
  -> validação e validação cruzada
  -> avaliação final única no teste
  -> HTML autocontido

Novo CSV no navegador
  -> leitura local progressiva
  -> escolha explícita de colunas e rótulos
  -> validação
  -> EDA local
  -> atualização dos componentes visuais
```

## EDA e qualidade

O relatório apresentará, no mínimo:

- número de linhas e colunas;
- proporção das classes;
- ausências e linhas inválidas;
- distribuição do tamanho dos textos em caracteres e palavras;
- presença de marcação HTML;
- quantidade de textos duplicados;
- comparação do comprimento dos textos entre classes;
- amostra pequena e determinística de resenhas de cada classe.

O dataset atual possui 50.000 linhas, 25.000 exemplos por classe, nenhuma resenha vazia, 29.200 textos com `<br />` e 406 grupos de textos duplicados, abrangendo 824 linhas. Esses números deverão ser recalculados pelo pipeline, não codificados como constantes.

## Limpeza de texto

A rotina seguirá esta ordem:

1. remover marcação HTML simples e decodificar entidades HTML;
2. converter o texto para minúsculas;
3. remover pontuação, números e caracteres especiais;
4. normalizar espaços;
5. remover stopwords em inglês.

As negações `not`, `no`, `nor` e `never` serão preservadas, pois carregam sinal de sentimento. Stemming e lematização serão omitidos por serem opcionais e adicionarem custo e dependências sem benefício garantido para os modelos lineares escolhidos.

## Deduplicação e divisão

Textos exatamente iguais serão agrupados antes da divisão. Como os grupos duplicados encontrados no dataset atual não apresentam conflito de rótulo, será mantida uma ocorrência por texto.

Os dados únicos serão divididos de forma estratificada em:

- 70% para treino;
- 15% para validação;
- 15% para teste.

A semente será fixa em `42`. Após o split, uma verificação deverá confirmar que nenhum texto aparece em mais de um conjunto.

## Representação e modelos

A representação escolhida é Bag of Words, implementada com `CountVectorizer`:

- unigramas e bigramas;
- `min_df=2`;
- limite inicial de 50.000 características;
- vocabulário e transformações ajustados apenas no treino.

Serão comparados dois pipelines fixos:

1. `MultinomialNB` como baseline;
2. `LogisticRegression` como segundo modelo.

Os dois modelos serão avaliados na validação. O candidato com maior F1 será selecionado antes de revelar o teste e passará por validação cruzada estratificada de cinco folds usando somente os dados de treino. Depois de congeladas as decisões, ambos os modelos serão avaliados uma única vez no teste para atender à comparação solicitada, sem novo ajuste baseado nesses resultados.

## Métricas e interpretação

Para cada modelo, o HTML exibirá:

- accuracy;
- precision;
- recall;
- F1 da classe positiva;
- matriz de confusão.

O relatório também mostrará uma amostra determinística de falsos positivos e falsos negativos, com texto truncado apenas para apresentação, além de uma discussão curta sobre negação, ironia, ambiguidade, vocabulário raro e resenhas mistas. Nenhuma conclusão sobre causas será apresentada sem exemplos observados.

## Relatório HTML

O relatório será uma página única, responsiva e autocontida. Não dependerá de internet, API ou servidor para abrir. O dataset bruto não será incorporado ao arquivo; apenas agregados, métricas e pequenas amostras necessárias à apresentação.

Seções previstas:

1. resumo executivo;
2. objetivo e metodologia;
3. jornada animada do pipeline;
4. EDA e qualidade dos dados;
5. limpeza, split e prevenção de vazamento;
6. comparação dos modelos;
7. erros observados;
8. decisões técnicas e justificativas;
9. conclusões e próximos passos;
10. análise local de outro CSV.

As animações incluirão progresso real de leitura do arquivo, avanço entre etapas, contadores e barras dos gráficos. O relatório respeitará `prefers-reduced-motion` e continuará compreensível com animações desativadas.

## Análise de novos CSVs no navegador

O navegador lerá o arquivo progressivamente para evitar carregar todo o conteúdo como uma única string. Após reconhecer o cabeçalho, a interface solicitará explicitamente:

- coluna que contém o texto;
- coluna que contém a classe;
- valor que significa classe positiva;
- valor que significa classe negativa.

Somente depois dessas escolhas a EDA será executada. Ela mostrará linhas válidas e inválidas, classes, ausências, duplicatas, marcação HTML e distribuição aproximada de tamanho dos textos. Classes ausentes, escolhas iguais, colunas inexistentes ou CSV ilegível devem gerar mensagens acionáveis. Essa área não aplicará o modelo do IMDb nem treinará outro modelo.

## Estrutura prevista

```text
src/build_report.py
src/report_template.html
src/vendor/papaparse.min.js
tests/test_pipeline.py
requirements.txt
reports/imdb_analysis.html
docs/superpowers/specs/2026-09-09-imdb-sentiment-analysis-design.md
```

`build_report.py` concentrará o fluxo em funções pequenas de validação, preparação, modelagem e renderização. O template conterá apenas apresentação e EDA executada no navegador. Papa Parse será incorporado ao HTML gerado para leitura correta de CSVs locais; os gráficos simples usarão HTML, CSS e JavaScript nativos, evitando outra biblioteca visual.

## Verificação

Um único arquivo de testes, usando dados sintéticos pequenos, deverá verificar:

- limpeza e preservação das negações;
- rejeição de esquema ou rótulos inválidos;
- deduplicação anterior ao split;
- estratificação e ausência de sobreposição;
- inexistência de vocabulário aprendido fora do treino;
- presença das seções obrigatórias no HTML gerado.

Depois dos testes sintéticos, o pipeline será executado integralmente sobre as 50.000 resenhas. A entrega só será considerada pronta se os testes terminarem com código de saída zero, o HTML for gerado e as métricas exibidas corresponderem aos artefatos calculados nessa execução.

## Critérios de aceite

- O HTML abre localmente por duplo clique e não faz requisições de rede.
- O estudo do IMDb cobre todas as partes obrigatórias do formulário.
- O CSV original permanece inalterado e fora do Git.
- Duplicatas não atravessam os conjuntos de treino, validação e teste.
- O conjunto de teste não influencia limpeza, vocabulário, escolha ou ajuste dos modelos.
- Os dois modelos possuem métricas comparáveis e matriz de confusão.
- O relatório explica decisões e limitações em linguagem adequada para uma entrevista.
- Um novo CSV pode ser selecionado e analisado localmente após escolhas explícitas de colunas e rótulos.
- Falhas de entrada aparecem como mensagens claras, sem resultados silenciosamente incorretos.
