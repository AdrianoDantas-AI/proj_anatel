# README técnico Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Documentar em português como executar, entender e acessar o relatório publicado do projeto.

**Architecture:** Um único `README.md` descreve o pipeline existente e aponta para o relatório publicado no GitHub Pages. Não altera código, dados, dependências nem o deploy.

**Tech Stack:** Markdown, Python, pytest e GitHub Pages.

---

## Estrutura de arquivos

- Criar: `README.md` — documentação técnica pública do projeto.
- Verificar: `src/build_report.py` — interface de linha de comando documentada.
- Verificar: `tests/test_pipeline.py` — suíte de testes documentada.

### Task 1: Escrever e verificar o README

**Files:**

- Create: `README.md`
- Verify: `src/build_report.py:487-526`
- Verify: `tests/test_pipeline.py`

- [ ] **Step 1: Confirmar a interface documentada**

Run: `& .venv\\Scripts\\python.exe src\\build_report.py --help`

Expected: exit code `0` e opções `--input`, `--text-column`, `--label-column`, `--positive-label`, `--negative-label` e `--output`.

- [ ] **Step 2: Criar `README.md`**

O arquivo deve conter: objetivo; link para `https://adrianodantas-ai.github.io/proj_anatel/`; fluxo de dados; decisões de modelagem e controles contra vazamento; requisitos; instalação; formato e comando de execução do CSV; testes; estrutura de arquivos; limitações do escopo.

- [ ] **Step 3: Verificar o Markdown e os comandos**

Run: `git diff --check && & .venv\\Scripts\\python.exe -m pytest`

Expected: exit code `0` e 13 testes aprovados.

- [ ] **Step 4: Commit e publicação**

Run: `git add README.md docs/superpowers/plans/2026-09-09-readme-tecnico.md; git commit -m "docs: add technical project readme"; git push origin main`

Expected: exit code `0` e README publicado no branch `main`.
