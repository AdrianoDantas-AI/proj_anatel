# README técnico — Design

**Data:** 2026-09-09  
**Status:** aprovado para implementação

## Objetivo

Adicionar um `README.md` em português que permita entender, instalar, executar e verificar o estudo de análise de sentimentos IMDb sem ler o código-fonte.

## Conteúdo

O README deverá apresentar:

- objetivo e resultado gerado pelo pipeline;
- fluxo técnico, da validação do CSV ao relatório HTML autocontido;
- decisões de modelagem: Bag of Words com unigramas e bigramas, Naive Bayes, Regressão Logística, negações preservadas e experimento de normalização;
- controles contra vazamento: deduplicação antes do split, vetorizador ajustado no treino e teste consultado após as decisões;
- requisitos, instalação, formato esperado do CSV, comando de geração e comando de testes;
- estrutura de arquivos e limites explícitos do escopo.

## Limites

- O README não incluirá dataset, métricas fixas, relatório gerado nem imagens.
- Não haverá alterações de código, dependências ou comportamento.
- Os comandos documentados refletirão a interface atual de `src/build_report.py` e a suíte `pytest`.

## Verificação

Antes da entrega, verificar que os comandos citados aceitam `--help` e que a suíte de testes passa no checkout alterado.
