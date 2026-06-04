# Referencias - SCC0633-SCC5908 PLN

## Bases de dados

- **B2W-Reviews01** — Real, L.; Oshiro, M.; Mafra, A. B2W-Reviews01: an open product reviews corpus. 2019. Disponivel em: https://github.com/b2wdigital/b2w-reviews01
- **Olist Brazilian E-Commerce Public Dataset** — Olist. Brazilian E-Commerce Public Dataset by Olist. Kaggle, 2018. Disponivel em: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **Mercado Livre API** — Mercado Livre. Documentacao da API de opinioes sobre um produto. Disponivel em: https://developers.mercadolivre.com.br/pt_br/opinioes-sobre-um-produto

## Corpus gold (Etapa 2)

- **SentiBR** — Brum, H.B.; Nunes, M.G.V. Building a Sentiment Corpus of Tweets in Brazilian Portuguese. arXiv:1712.08917, 2017. Disponivel em: https://github.com/edilsonacjr/sentibr
- **ReLi** (alternativa) — corpus de resenhas literarias em portugues com anotacao de sentimento por sentenca. Disponivel em: https://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/reli/

## Lexicos externos (Etapa 2)

- **OpLexicon v3.0** — Souza, M.; Vieira, R.; Busetti, F.; Chishman, R.; Alves, I.M. Construction of a Portuguese Opinion Lexicon from Multiple Reviews Types. In: 8th Brazilian Symposium in Information and Human Language Technology (STIL), 2011. Disponivel em: http://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/oplexicon/
- **SentiLex-PT 02** — Carvalho, P.; Sarmento, L.; Teixeira, J.; Silva, M.J. SentiLex-PT: Main Characteristics and Challenges. In: 10th International Conference on the Computational Processing of Portuguese (PROPOR), 2012. Disponivel em: http://l2f.inesc-id.pt/wiki/index.php/SentiLex-PT

## Algoritmos de PLN (Etapa 2)

- **RSLP (Removedor de Sufixos da Lingua Portuguesa)** — Orengo, V.M.; Huyck, C. A Stemming Algorithm for the Portuguese Language. In: Proceedings of the Eighth International Symposium on String Processing and Information Retrieval (SPIRE). IEEE, 2001. pp. 186-193.
- **spaCy pt_core_news_sm** — Honnibal, M.; Montani, I. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. 2017. Disponivel em: https://spacy.io

## Tecnicas gerais

- Analise de sentimentos
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Regressao Logistica
- Linear SVC (Support Vector Classification)
- Regras simbolicas com lexicos e padroes discursivos
- Recomendacao baseline baseada em sentimento

## Documentos internos

- `docs/etapa2_plano.md` — plano de melhorias da Etapa 2
- `docs/relatorio_final.md` — relatorio consolidado da ultima execucao
- `docs/relatorio_solucao_simbolica.md` — documentacao do analisador simbolico
- `docs/regras_simbolicas_viabilidade.md` — analise de viabilidade das regras
- `docs/codigo_comentado.md` — organizacao do codigo
- `docs/coleta_base_real_simples.md` — metodologia de coleta do Mercado Livre
- `docs/resposta_comentarios_professor.md` — respostas aos comentarios da segunda entrega
