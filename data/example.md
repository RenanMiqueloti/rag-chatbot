# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation, ou RAG, é um padrão de arquitetura que combina busca em uma base de conhecimento externa com geração de texto por LLMs. Em vez de depender apenas do que o modelo memorizou durante o treinamento, o sistema busca trechos relevantes em documentos próprios e os fornece como contexto para a resposta.

## Por que RAG existe

LLMs treinados em dados públicos têm dois limites práticos. O primeiro é a defasagem temporal: o conhecimento congela na data do corte do treinamento. O segundo é a ausência de informação proprietária — nenhum modelo público viu o manual interno da sua empresa, o histórico de tickets de suporte ou a documentação técnica do produto.

Fine-tuning resolve parcialmente, mas custa caro, exige dados rotulados e fica obsoleto a cada nova versão do conteúdo. RAG resolve sem treinamento: basta indexar os documentos e o pipeline busca o que importa em tempo de consulta.

## Como funciona em alto nível

O pipeline tem três etapas:

1. **Retrieve.** Dado o texto da pergunta, o sistema procura os trechos mais relevantes na base. Pode usar similaridade vetorial (embeddings), busca por palavra-chave (BM25), ou ambos combinados.
2. **Augment.** Os trechos recuperados são inseridos como contexto no prompt enviado ao LLM, geralmente com uma instrução do tipo "responda usando apenas o contexto abaixo".
3. **Generate.** O LLM gera a resposta condicionada ao contexto. Se o contexto não cobre a pergunta, modelos bem instruídos respondem que não sabem em vez de inventar.

## Componentes técnicos comuns

- **Embeddings.** Modelos que convertem texto em vetores. `sentence-transformers/all-MiniLM-L6-v2` é leve e cabe em CPU; modelos multilíngues como `paraphrase-multilingual-MiniLM-L12-v2` funcionam melhor em português.
- **Vector store.** Banco que armazena vetores e responde "quais são os K mais próximos deste vetor de consulta". Qdrant, Pinecone, Weaviate, Milvus e FAISS são opções comuns.
- **BM25.** Algoritmo de ranking por palavra-chave que dá peso a termos raros e exatos. Captura siglas, IDs e nomes próprios que embeddings deixam passar.
- **Reciprocal Rank Fusion (RRF).** Estratégia para fundir múltiplas listas de ranking sem precisar calibrar pesos.
- **Reranker.** Cross-encoder que reordena os candidatos do retrieval com base na query inteira. Melhora a precisão no topo do ranking.

## Trade-offs

RAG não substitui um modelo bom. Se o LLM base alucina muito ou não segue instruções, juntar contexto não corrige. RAG também não cobre raciocínio complexo que exige sintetizar informação fragmentada em vários documentos — para esses casos, agentes com múltiplos passos costumam ser melhores.

A qualidade do retrieval é o gargalo mais comum. Embeddings que não falam o idioma do corpus, chunks mal cortados, ou ausência de BM25 são causas frequentes de respostas ruins. Quando o sistema responde "não sei" a perguntas claramente cobertas pelo documento, o problema raramente é o LLM — é o que chega até ele no contexto.

## Onde RAG é útil

Atendimento ao cliente sobre produto específico, busca em base interna de engenharia, análise de documentos jurídicos ou contratos, leitura de relatórios financeiros, suporte técnico baseado em manuais. Qualquer caso em que existe um corpus delimitado e o usuário faz perguntas em linguagem natural sobre ele.
