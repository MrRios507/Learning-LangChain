# SelfQueryRetriever: Cómo Transformar Consultas en Lenguaje Natural en Filtros Estructurados con LangChain

## Introducción

En el mundo de las aplicaciones RAG (Retrieval-Augmented Generation), una de las mayores limitaciones es que la búsqueda pura por similitud semántica no siempre entrega resultados precisos. Un usuario que busca "películas de ciencia ficción estrenadas después de 2010 con rating superior a 8" obtendrá resultados que *parecen* relevantes semánticamente, pero pueden incluir películas de 1985 o con rating de 4.0.

La solución es combinar **búsqueda semántica** con **filtros de metadatos estructurados**. Pero construir esos filtros manualmente para cada consulta del usuario es tedioso e inviable en producción.

Aquí es donde entra **`SelfQueryRetriever`** de LangChain: un componente que utiliza un LLM para *traducir automáticamente* consultas en lenguaje natural a filtros de metadatos ejecutables.

---

## ¿Qué es el SelfQueryRetriever?

`SelfQueryRetriever` es un retriever que, como su nombre sugiere, **se consulta a sí mismo**. Dada una consulta en lenguaje natural, hace lo siguiente:

1. **Usa un LLM** para extraer filtros de metadatos de la consulta del usuario.
2. **Construye una query estructurada** que combina la búsqueda semántica (similitud de embeddings) con filtros sobre metadatos.
3. **Ejecuta esa query** contra el vector store subyacente.
4. **Retorna los documentos** que cumplen ambos criterios: similitud semántica Y filtros de metadata.

En esencia, el LLM actúa como un **parseador inteligente** que entiende la intención del usuario y la traduce a operaciones de filtrado que el vector store puede ejecutar eficientemente.

---

## Arquitectura Interna

```
┌─────────────────────────────────────────────────────┐
│                  Usuario                             │
│   "Películas de ciencia fiction con rating > 8.5"   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              SelfQueryRetriever                      │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │  LLM Chain   │───▶│  Query Constructor       │   │
│  │  (parseo)    │    │  - filter: rating > 8.5  │   │
│  └──────────────┘    │  - query: "ciencia       │   │
│                      │    ficción"               │   │
│                      └──────────┬───────────────┘   │
│                                 │                    │
│                      ┌──────────▼───────────────┐   │
│                      │  Structured Query        │   │
│                      │  Translator             │   │
│                      │  (convierte a sintaxis   │   │
│                      │   del vector store)      │   │
│                      └──────────┬───────────────┘   │
└─────────────────────────────────┼───────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────┐
│              Vector Store                           │
│   PGVector / Chroma / Pinecone / Weaviate / etc.    │
│                                                      │
│   Búsqueda: embedding_similarity AND metadata_filter │
└─────────────────────────────────────────────────────┘
```

---

## Componentes Clave

### 1. AttributeInfo: El Esquema de Metadatos

Antes de crear el retriever, necesitas describir **qué metadatos existen** en tus documentos. Esto se hace con `AttributeInfo`:

```python
from langchain_classic.chains.query_constructor.base import AttributeInfo

fields = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="A 1-10 rating for the movie",
        type="float",
    ),
]
```

Cada `AttributeInfo` tiene tres campos:

| Campo | Descripción |
|-------|-------------|
| `name` | Nombre del campo metadata (debe coincidir con lo que almacenaste) |
| `description` | Descripción en inglés para que el LLM entienda qué representa |
| `type` | Tipo de dato: `string`, `integer`, `float`, `string or list[string]` |

**Tip importante:** Las descripciones son críticas. El LLM las usa para decidir qué filtros aplicar. Describe el campo de forma clara y concisa.

### 2. Document Description

Una descripción breve de qué representan tus documentos:

```python
description = "Brief summary of a movie"
```

Esto le da al LLM contexto sobre el dominio de los datos.

### 3. LLM

El modelo de lenguaje que parseará las consultas:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="granite4.1:3b", temperature=0)
```

**Recomendaciones:**
- Usa `temperature=0` para respuestas deterministas.
- Modelos como GPT-4, Claude, o modelos locales como Llama 3 funcionan bien.
- Modelos más pequeños (3B-7B) pueden manejar tareas simples de parsing.

### 4. Vector Store

Cualquier vector store que soporte **metadata filtering**:

```python
from langchain_postgres import PGVector

vectorstore = PGVector.from_documents(
    docs,
    embeddings_model,
    connection=connection,
    collection_name=collection_name,
)
```

**Vector stores compatibles con metadata filtering:**
- PGVector (PostgreSQL)
- Chroma
- Pinecone
- Weaviate
- Qdrant
- Elasticsearch
- MongoDB Atlas

### 5. SelfQueryRetriever

El componente que une todo:

```python
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    description,
    fields
)
```

---

## Flujo de Ejemplo Completo

### Paso 1: Preparar documentos con metadata

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={"year": 1979, "director": "Andrei Tarkovsky", "genre": "thriller", "rating": 9.9},
    ),
]
```

### Paso 2: Crear el vector store

```python
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

embeddings_model = OllamaEmbeddings(model="embeddinggemma")

vectorstore = PGVector.from_documents(
    docs,
    embeddings_model,
    connection="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
    collection_name="movie_reviews",
)
```

### Paso 3: Configurar el SelfQueryRetriever

```python
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_ollama import ChatOllama

fields = [
    AttributeInfo(name="genre", description="The genre of the movie", type="string or list[string]"),
    AttributeInfo(name="year", description="The year the movie was released", type="integer"),
    AttributeInfo(name="director", description="The name of the movie director", type="string"),
    AttributeInfo(name="rating", description="A 1-10 rating for the movie", type="float"),
]

description = "Brief summary of a movie"
llm = ChatOllama(model="granite4.1:3b", temperature=0)
retriever = SelfQueryRetriever.from_llm(llm, vectorstore, description, fields)
```

### Paso 4: Consultar

```python
# Consulta con filtro de rating
results = retriever.invoke("I want to watch a movie rated higher than 8.5")
# → El LLM genera: filter = rating > 8.5

# Consulta con múltiples filtros
results = retriever.invoke("What's a highly rated (above 8.5) science fiction film?")
# → El LLM genera: filter = rating > 8.5 AND genre = "science fiction"

# Consulta solo semántica (sin filtros explícitos)
results = retriever.invoke("movies about dinosaurs")
# → Solo búsqueda por similitud semántica
```

---

## Qué Hace el LLM Internamente

Cuando el usuario escribe: *"I want to watch a movie rated higher than 8.5"*

El LLM genera internamente una **StructuredQuery** como esta:

```json
{
  "query": "movies rated higher than 8.5",
  "filter": {
    "type": "comparison",
    "attribute": "rating",
    "operator": "$gt",
    "value": 8.5
  }
}
```

Luego, el **Structured Query Translator** convierte esto a la sintaxis nativa del vector store. Para PGVector sería algo como:

```python
vectorstore.similarity_search(
    query="movies rated higher than 8.5",
    filter={"rating": {"$gt": 8.5}}
)
```

---

## Operadores de Filtro Soportados

Los filtros que el LLM puede generar dependen del vector store. Los operadores más comunes son:

### Comparación
| Operador | Descripción | Ejemplo |
|----------|-------------|---------|
| `$eq` | Igual a | `genre = "sci-fi"` |
| `$ne` | No igual a | `genre != "comedy"` |
| `$gt` | Mayor que | `rating > 8.0` |
| `$gte` | Mayor o igual | `rating >= 8.0` |
| `$lt` | Menor que | `year < 2000` |
| `$lte` | Menor o igual | `year <= 2000` |

### Lógicos
| Operador | Descripción |
|----------|-------------|
| `$and` | Ambas condiciones verdaderas |
| `$or` | Al menos una condición verdadera |
| `$not` | Niega la condición |

### Arrays
| Operador | Descripción |
|----------|-------------|
| `$in` | El valor está en la lista |
| `$nin` | El valor no está en la lista |

---

## Soporte por Vector Store

| Vector Store | Metadata Filtering | Operadores soportados |
|-------------|-------------------|----------------------|
| **PGVector** | ✅ | `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$and`, `$or`, `$not` |
| **Chroma** | ✅ | `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$and`, `$or` |
| **Pinecone** | ✅ | `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`, `$not` |
| **Weaviate** | ✅ | `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$and`, `$or` |
| **Qdrant** | ✅ | `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$and`, `$or`, `$not` |
| **Elasticsearch** | ✅ | `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$and`, `$or`, `$not` |

> **Nota:** No todos los vector stores soportan los mismos operadores. Verifica la documentación del tuyo.

---

## SelfQueryRetriever vs Filtrado Manual

### Filtrado Manual (sin SelfQueryRetriever)

```python
# El desarrollador debe escribir la lógica de parsing
def parse_query(user_query: str) -> dict:
    if "rated higher than" in user_query:
        rating = extract_number(user_query)
        return {"rating": {"$gt": rating}}
    elif "science fiction" in user_query:
        return {"genre": "science fiction"}
    # ... decenas de casos más
    return {}

filter_expr = parse_query(user_query)
results = vectorstore.similarity_search(query=user_query, filter=filter_expr)
```

**Problemas:**
- Lógica personalizada frágil y difícil de mantener.
- No escala a muchos campos de metadata.
- No entiende variaciones del lenguaje natural.

### Con SelfQueryRetriever

```python
retriever = SelfQueryRetriever.from_llm(llm, vectorstore, description, fields)
results = retriever.invoke(user_query)
```

**Ventajas:**
- El LLM entiende variaciones del lenguaje natural.
- Maneja múltiples filtros automáticamente.
- Escalable: agregar un nuevo campo metadata solo requiere un nuevo `AttributeInfo`.
- Funciona con consultas ambiguas o complejas.

---

## Casos de Uso Reales

### 1. E-commerce
*"Muéstrame productos electrónicos de Apple con precio menor a $500 y rating superior a 4.5"*

Filtros generados: `category = "electronics" AND brand = "Apple" AND price < 500 AND rating > 4.5`

### 2. Documentos Legales
*"Contratos vigentes firmados después de 2024 con cláusula de confidencialidad"*

Filtros generados: `status = "active" AND signed_date > 2024 AND has_confidentiality = true`

### 3. Noticias/Artículos
*"Artículos de tecnología publicados en los últimos 30 días con más de 1000 visitas"*

Filtros generados: `category = "technology" AND date > (hoy - 30 días) AND views > 1000`

### 4. Soporte al Cliente
*"Tickets urgentes del departamento de facturación abiertos esta semana"*

Filtros generados: `priority = "urgent" AND department = "billing" AND status = "open" AND created_date > (hoy - 7 días)`

---

## Mejores Prácticas

### 1. Descripciones Claras de Metadatos

```python
# ❌ Malo
AttributeInfo(name="yr", description="yr", type="int")

# ✅ Bueno
AttributeInfo(
    name="year",
    description="The year the movie was released (e.g., 1993, 2010)",
    type="integer"
)
```

### 2. Usa temperature=0

El LLM debe ser determinista para parsing de queries. Nunca uses temperatura alta.

### 3. Valida los Filtros Generados

```python
# Habilita verbose para debugging
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, description, fields,
    verbose=True  # Muestra la query estructurada generada
)
```

### 4. Limita los Campos de Metadata

No incluyas campos que no necesites filtrar. Menos campos = mejor rendimiento del LLM.

### 5. Considera Few-Shot Learning

Si el LLM no genera filtros correctos, puedes proporcionar ejemplos en la descripción:

```python
# Algunos retrievers permiten pasar ejemplos
chain_kwargs={
    "examples": [
        ("movies rated higher than 8.5", {"filter": {"rating": {"$gt": 8.5}}}),
        ("science fiction films from the 90s", {"filter": {"genre": "science fiction", "year": {"$gte": 1990, "$lt": 2000}}}),
    ]
}
```

---

## Errores Comunes

### 1. Usar un LLM que no soporta Function Calling

Algunos modelos pequeños no generan JSON válido. Usa modelos probados o verifica el output.

### 2. Metadata inconsistente

Si un documento tiene `metadata={"year": "1993"}` (string) y otro `metadata={"year": 1993}` (int), los filtros numéricos fallarán.

### 3. Olvidar instalar `lark`

SelfQueryRetriever requiere el paquete `lark` para parsear queries:

```bash
pip install lark
```

### 4. No documentar todos los campos

Si tu metadata tiene un campo `director` pero no lo incluyes en `fields`, el LLM no podrá filtrar por director.

---

## Dependencias

```bash
pip install langchain-classic langchain-core langchain-ollama langchain-postgres lark
```

Para otros vector stores:

```bash
# Chroma
pip install langchain-chroma

# Pinecone
pip install langchain-pinecone

# Weaviate
pip install langchain-weaviate

# Qdrant
pip install langchain-qdrant
```

---

## Conclusión

`SelfQueryRetriever` resuelve uno de los problemas más comunes en RAG: **cómo hacer que la búsqueda sea tanto semántica como estructurada**. Al delegar la extracción de filtros a un LLM, eliminamos la necesidad de escribir parsing manual para cada tipo de consulta, making our RAG systems more intelligent, scalable, and user-friendly.

La combinación de:
- **Búsqueda semántica** (entender la intención)
- **Filtros de metadatos** (restringir por atributos concretos)
- **LLM como traductor** (lenguaje natural → filtros estructurados)

...crea un sistema de recuperación que realmente entiende lo que el usuario quiere.

---

## Referencias

- [LangChain Self Querying Retrieval - Docs Oficiales](https://python.langchain.com/docs/how_to/self_query/)
- [SelfQueryRetriever Reference](https://reference.langchain.com/python/langchain-classic/retrievers/self_query/base/SelfQueryRetriever)
- [Enhancing RAG Performance with Metadata - Medium](https://medium.com/@lorevanoudenhove/enhancing-rag-performance-with-metadata-the-power-of-self-query-retrievers-e29d4eecdb73)
- [Elasticsearch Self-Query Retriever Tutorial](https://www.elastic.co/search-labs/tutorials/examples/self-query-retriever-langchain-elasticsearch-chatbot)
- [Metadata Filtering in Vector Search - Comprehensive Guide](https://www.saumilsrivastava.ai/blog/metadata-filtering-in-vector-search-a-comprehensive-guide-for-engineering-leaders)
