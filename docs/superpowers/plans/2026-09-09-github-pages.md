# GitHub Pages do relatório IMDb Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publicar somente o relatório HTML versionado na raiz do site GitHub Pages do repositório.

**Architecture:** O workflow prepara `_site/index.html` a partir de `reports/imdb_analysis.html` e publica exclusivamente esse diretório como artefato Pages. O job de deploy consome o artefato após o build, preservando o gatilho em `main`, a execução manual e a proteção de concorrência.

**Tech Stack:** GitHub Actions, GitHub Pages, Bash do `ubuntu-latest`.

---

## Estrutura de arquivos

- Modificar: `.github/workflows/static.yml` — preparar e publicar o artefato estático mínimo.
- Verificar: `reports/imdb_analysis.html` — fonte HTML versionada que será entregue como `index.html`.

### Task 1: Restringir o artefato do Pages ao relatório

**Files:**

- Modify: `.github/workflows/static.yml`
- Verify: `reports/imdb_analysis.html`

- [ ] **Step 1: Confirmar a fonte do artefato**

Run: `test -f reports/imdb_analysis.html`

Expected: exit code `0`.

- [ ] **Step 2: Substituir o conteúdo de `.github/workflows/static.yml`**

```yaml
name: Deploy static content to Pages

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v6
      - name: Setup Pages
        uses: actions/configure-pages@v5
      - name: Prepare report
        run: |
          test -f reports/imdb_analysis.html
          mkdir -p _site
          cp reports/imdb_analysis.html _site/index.html
          touch _site/.nojekyll
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v4
        with:
          path: _site

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 3: Revisar a alteração de workflow**

Run: `git diff --check && git diff -- .github/workflows/static.yml`

Expected: exit code `0`; o único caminho passado a `upload-pages-artifact` é `_site`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/static.yml
git commit -m "fix: publish report through GitHub Pages"
```

### Task 2: Verificar o deploy publicado

**Files:**

- Verify: `.github/workflows/static.yml`
- Verify: `reports/imdb_analysis.html`

- [ ] **Step 1: Executar os testes locais existentes**

Run: `python -m pytest`

Expected: exit code `0`.

- [ ] **Step 2: Enviar o commit para `origin/main`**

Run: `git push origin main`

Expected: exit code `0` e criação de uma execução `Deploy static content to Pages`.

- [ ] **Step 3: Acompanhar a execução de Pages**

Run: `gh run list --workflow static.yml --branch main --limit 1 --json databaseId,status,conclusion --jq '.[0]'`

Expected: `status` igual a `completed` e `conclusion` igual a `success`.

- [ ] **Step 4: Verificar a URL pública**

Run: `curl.exe -L -s -o NUL -w "HTTP %{http_code}\n" https://adrianodantas-ai.github.io/proj_anatel/`

Expected: `HTTP 200`.
