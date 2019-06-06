import requests
import ndjson


class Engine(object):
    def __init__(self, host):
        self.session = requests.Session()
        self.host = host

    def search(self, json_query):
        r = self.session.get(self.host + "/_search", headers={"Content-Type": "application/json"},
                         data=json_query)
        return r.json()
    
    def msearch(self, json_queries, *args, **kwargs):
        r = self.session.get(self.host + "/_msearch", headers={"Content-Type": "application/x-ndjson"},
                    data=ndjson.dumps(json_queries) + "\n")
        return r.json()
