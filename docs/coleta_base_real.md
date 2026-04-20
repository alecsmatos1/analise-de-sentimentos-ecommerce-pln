# Coleta de Base Real Complementar

## Fonte

- Mercado Livre Brasil.
- Reviews coletadas pela API oficial de opiniões de produto.
- Descoberta de itens feita exclusivamente por endpoints oficiais de catálogo e produtos.

## Critérios escolhidos para reduzir viés

- Usar apenas a API oficial como fonte dos textos das reviews.
- Tratar a base como complementar, nunca como base principal.
- Selecionar itens com distribuição mista de notas.
- Exigir pelo menos `30` reviews com comentário por item.
- Exigir pelo menos `1` avaliações negativas agregadas (1-2 estrelas) por item.
- Exigir pelo menos `5` avaliações positivas agregadas (4-5 estrelas) por item.
- Limitar a coleta a no máximo `100` reviews por item para evitar concentração excessiva.
- Usar múltiplos termos de busca para espalhar a amostragem por diferentes domínios de consumo.

## Procedimento para refazer

1. Definir a variável de ambiente `MELI_ACCESS_TOKEN` com um token válido do app.
2. Executar `python collect_meli_reviews.py` no diretório do projeto.
3. Ler o arquivo `data/mercadolivre_reviews.csv` gerado ao final.

## Observações de conformidade

- A coleta usou a API oficial tanto para descobrir os itens quanto para obter os textos das reviews.
- Os dados foram coletados para uso acadêmico complementar.
- O arquivo gerado não deve ser republicado como espelho bruto sem revisão dos termos da plataforma.

## Resumo da execução

- Data da coleta: 2026-04-19 21:42:34
- Itens candidatos encontrados: 162
- Itens inspecionados na API: 162
- Itens selecionados: 12
- Reviews coletadas: 921

## Distribuição das notas na base coletada

- 1 estrela: 85
- 2 estrelas: 16
- 3 estrelas: 23
- 4 estrelas: 70
- 5 estrelas: 727

## Limitações

- A seleção dos textos depende do que a API oficial expõe para cada item.
- Mesmo com filtros de distribuição, a base complementar ainda pode refletir o viés do ecossistema de avaliações da plataforma.

## Itens selecionados

item_id | product_id | item_title | category_id | reviews_with_comment | rating_average
--- | --- | --- | --- | --- | ---
MLB6586337520 | MLB54520373 | Maleta Para Notebook, Bolsa Notebook Para 15,6 Polegadas Masculina Impermeável E Acolchoada, Pasta Para Notebook Executiva Portátil Estilo Bandolera Acessórios Para Notebook Taygeer 1073 Preto Marron | MLB16547 | 324 | 4.9
MLB4469488139 | MLB38770647 | Maleta Para Notebook, Pasta Para Notebook De 17 Polegadas Masculina Executiva Portátil Expansível, Bolsa Notebook Impermeável E Acolchoada Com Alça Transversal, Mancro 1084 Preto | MLB16547 | 148 | 4.9
MLB4456589973 | MLB25135551 | Relogio Inteligente Smartwatch D20 Preto + Fone Bluetooth Redmi | MLB135384 | 35 | 4.2
MLB3362036753 | MLB15583423 | Smartphone Multi F Pro Preto - P9118 | MLB1055 | 174 | 3.6
MLB1964174846 | MLB14982892 | Smartphone Motorola One Vision 128gb Dual Bronze | MLB1055 | 69 | 4.4
MLB5991065346 | MLB16206437 | Smartphone Multilaser E Lite 32gb Preto P9126 | MLB1055 | 564 | 3.7
MLB4469262371 | MLB41629336 | Smartphone Xiaomi Redmi 13 Cor Azul-marinho | MLB1055 | 775 | 4.6
MLB5396148200 | MLB50178919 | Kaisasa Shampoo Tonalizante Para Cobrir Cabelo Cinzento，shampoo cabelo branco，shampoo barba，shampoo redutor grisalhos，shampoo tonalizante preto，hair dye shampoo，black hair dye，500ml-Preto | MLB1265 | 1018 | 4.8
MLB4543754693 | MLB52051148 | Meidu Shampoo Tonalizante Para Cobrir Cabelo Cinzento，shampoo black hair dye，shampoo grisalho，shampoo cabelo preto，shampoo para cabelos brancos，COLORAÇÃO EM SHAMPOO preto 500ml | MLB1265 | 232 | 4.8
MLB5454690706 | MLB22813112 | Cadeira Escritório Secretaria Rcp Preto | MLB193945 | 32 | 4.2
MLB6259414652 | MLB45283916 | Tênis Corrida Runfalcon 5 adidas | MLB23332 | 1672 | 4.8
MLB5850724132 | MLB29816505 | Mochila Yepp Escolar Mochila Infantil Feminina De Costas Menina Mochila Menina Mochila Infantil Feminina Mochila Panda Mochila Flamingo Mochila Lhama Cor Aranha Dourado 23l | MLB3127 | 181 | 4.8