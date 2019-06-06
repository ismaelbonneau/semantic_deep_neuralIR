# Elastic Search
## Installation
Follow [these](https://www.elastic.co/guide/en/elasticsearch/reference/current/zip-targz.html) instructions to install ElasticSearch.

## Kibana
Follow [these]() instructions to install Kibana.

Kibana is a tool to interact with ES. You can monitore and to requests to check whether it's working or not.

## Some config
If you're running your ES locally, there's nothing to configure. If not, you have to check the `elasticsearh.yml` and `kibana.yml` files. Check [this](https://www.elastic.co/guide/en/elasticsearch/reference/current/settings.html) and [this](https://www.elastic.co/guide/en/kibana/current/settings.html) to know where are the config files, depending on your installation.

## Using Kibana
Once you [started](https://www.elastic.co/guide/en/elasticsearch/reference/current/starting-elasticsearch.html) ElasticSearch, you can [start](https://www.elastic.co/guide/en/kibana/6.7/start-stop.html) Kibana. Once done, go on `localhost:5601` (or the server you ran Kibana on) and you should see Kibana running. The most important pannel to begin with is the Dev Tool's one. Just click on it and you can play with ES and copy-paste what's in the [doc](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html).

## Some tips to use ElasticSearch in Python
All the things incoming can be done using the Kibana Console. To discover it's probably easier to use it than Python's API. Python's API is great when you need to interract within a program with ES like when you will do the indexation of a lot of files. The main page for the "how-tos" are [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices.html) for creating an index, [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs.html) for documents and [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs.html) for searches. You might want to take a look at [this](https://www.elastic.co/guide/en/elasticsearch/reference/current/cat.html) one to get info from what you've done so far.

### Instalation
`pip3 install elasticsearch`

### Usage
Once you started ElasticSearch, you can instantiate an ElasticSearch object :

```python
import elasticsearch as es

engine = es.Elasticsearch()
```

To create an index :

```python
engine.indices.create(f"{index_name}", body=settings)
```

Here is a complete example

```python
index_name = "test-toto"
b = 0.5
k1 = 1
settings = {"settings": {
                "number_of_shards": 1,
                "index": {
                    "similarity": {
                        "default": {
                            "type":"BM25",
                            "b": b,
                            "k1": k1}
                        }
                    },},
                "mappings": {         
                    "trec": {
                        "properties": {
                            "title": {
                                "type": "text",
                                "index": "false"
                            },
                            "text": {
                                "type": "text",
                                "analyzer": "english"
                                }
                            }
                        }
                    }
                }
engine.indices.create(f"{index_name}", body=settings)
```

Using bulk to index multiple documents:
```python
import elasticsearch.helpers as helpers

def bulking(texts, titles):
    bulk_data = []
    for i, (txt, title) in enumerate(zip(texts, titles)):
        data_dict = {
                    '_index': 'test-toto',
                    '_type': 'trec',
                    '_id': str(i),
                    '_source': {
                        "text": txt,
                        "title": title
                    }
                }
        bulk_data.append(data_dict)
    return bulk_data

texts = ['toto is now gone',
         'toto has left the place',
         "toto went to his parent's place",
         "toto's place is quite far away"]

titles = ["Toto's life", 
          "Toto leaving",
          "Toto's trip",
          "Toto is far"]

helpers.bulk(engine, bulking(texts, titles))
```

Search example :
```python
engine.search("test-bm25", "trec", {"query": {"match": {"text": "toto place"}}})
```

Some useful commands (in the shell) :
```bash
# Show indices and their status, v for verbose
curl -XGET "localhost:9200/_cat/indices?v"

# Cat the mapping of trec type in robust2004 indice
curl "localhost:9200/robust2004/_mapping/trec?pretty"  # ?pretty for pretty print

# Test an analyzer
curl -XPOST "localhost:9200/_analyze?pretty" -H "Content-Type: application/json" -d '{"analyzer": "english", "text": "Any organisation has a responsability towards its employees, no matter how big it is."}'
```

If you read ES doc, a lot of examples are as follow :
```bash
POST _analyze
{
  "analyzer": "whitespace",
  "text":     "The quick brown fox."
}
```

If you're running this in local with a web server running, it might not work. That's why we use `curl`. When there's json arguments, you need the `-H` to specify it and the `d` to specify data. Writing all of these in one line is not easy. Thus you can type `\` at the end of a line and pressing `SUPER + ENTER` to jump a line. You can also prepare commands in files, making it easier. This leads to :

```bash
curl -XPOT \
"localhost:9200/_analyze?pretty" \
--header "Content-Type: application/json" \  # -H = --header
--data '{"analyzer": "english", "text": "Any organisation has a responsability towards its employees, no matter how big it is."}'  # -d = --data
```

If you need multi-line json, you can read [this](https://stackoverflow.com/questions/34847981/curl-with-multiline-of-json).