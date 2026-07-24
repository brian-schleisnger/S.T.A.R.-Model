agent/

&#x20; **\_\_init\_\_.py**

&#x09;this file contains nothing. \_\_init\_\_ files for python, however, should be in every folder. that way, other files see a package of code when going into that file, and it makes referencing a lot easier to understand.

&#x20; **cache.py**

&#x09;this is the primary long form memory system of the model. anytime a query is run, loop.py accesses cache.py at the beginning and at the end. once the question has ran and a response has been generated, loop then sends the question and response to cache. there, cache takes the response and stores the values as vectors (long strings of numbers). then, whenever a new question is run, loop goes and checks against cache's database. the new question is also embedded as a vector, and then that vector compares to all the other vectors in the cache. if there is one that is 92% similar (technically, if cosine similarity between the vectors is >.92) then the cache returns that similar response to loop, and loop skips the rest of it's steps and outputs that previous response. Note: the cache clears anytime you redeploy the app, meaning it's pretty temporary now, but as the model becomes more finalized, the cache will have longer windows to build up and be useful.

&#x20; **context.py**

&#x09;this is a pretty simple file supporting the inter-turn memory functions of 	

&#x20; loop.py

&#x20; memory.py

&#x20; schemas.py

dictionaries/

&#x20; acquisition\_data\_dictionary.json

&#x20; dish\_pl\_dictionary.json

&#x20; marketing\_spend\_dictionary.json

&#x20; sales\_dictionary.json

&#x20; subscriber\_count\_dictionary.json

Documentation/

&#x20; Data overview.txt

&#x20; extract\_docs.py

&#x20; extracted\_docstrings.md

&#x20; Goals.txt

&#x20; Idealized Final Product.txt

&#x20; Intern Project Scope.txt

&#x20; Tech stack specifics.txt

toolkit/

&#x20; \_\_init\_\_.py

&#x20; analytics.py

&#x20; base.py

&#x20; visuals.py

.gitattributes

.gitignore

app.py

app.yaml

logo1.png

requirements.txt

style.css

