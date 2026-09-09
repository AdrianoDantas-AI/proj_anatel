# GitHub Pages do relatório IMDb — Design

**Data:** 2026-09-09  
**Status:** aprovado para implementação

## Objetivo

Publicar o relatório HTML estático em `https://adrianodantas-ai.github.io/proj_anatel/` por GitHub Actions, sem expor o restante do repositório no artefato do Pages.

## Artefato

O relatório versionado em `reports/imdb_analysis.html` será copiado para um diretório temporário de publicação como `index.html`. Esse diretório, e somente ele, será enviado ao GitHub Pages.

O CSV de origem permanece fora do Git e não participa do workflow. Portanto, o workflow não gera o relatório: ele publica a versão já confirmada e versionada.

## Workflow

O workflow existente continuará sendo acionado por pushes para `main` e manualmente. Ele terá dois jobs:

1. `build`: faz checkout, configura Pages, cria o diretório de publicação, copia o relatório para `index.html` e envia esse diretório como artefato.
2. `deploy`: depende de `build`, usa o ambiente `github-pages` e publica o artefato.

As permissões necessárias ficam restritas a `contents: read`, `pages: write` e `id-token: write`. A concorrência permanece limitada a um deploy por vez.

## Limites

- Nenhum dado bruto, documentação interna, código-fonte ou arquivo de configuração será publicado como conteúdo do site.
- Não será criado site generator, dependência adicional ou etapa de treinamento no CI.
- O URL de projeto continuará com o sufixo `/proj_anatel/`.

## Verificação

- Validar a sintaxe YAML e a existência de `reports/imdb_analysis.html` antes do commit.
- Após o push, verificar execução bem-sucedida do workflow e HTTP 200 para a URL pública.
