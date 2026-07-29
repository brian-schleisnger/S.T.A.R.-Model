Tech stack specifics

* All of the files live in the Git. think of it like cloud storage designed for code
* You access the Git via GitHub.com. Each project has a repository ogn GitHub that contains everything.
* link to the GitHub code: https://github.com/brian-schleisnger/S.T.A.R.-Model.git
* you need to create a GitHub account and download GitHub desktop (no IT required)



* All the code was created and edited in VSCode. to use VSCode you have to get it through IT, search visual studio code and it should pop up.
* None of the extensions were super relevant to making the code
* Once we get access to ADK's (Agent Development Kits), examples being Google Antigravity, Cursor, Claude Code, or Codex, you could also build the code using that.
* a lot of these you can make free accounts - i did that primarily for long-term planning



* The LLM's used throughout the model are running through endpoints tied to databricks
* the code accesses the endpoint, the endpoint is tied to the AI gateway section of databricks lakebase, which is then connected to the model provider (anthropic, etc.)
* If you want to add new models, look at the ones you have access to on databricks
* Other platforms also have model endpoints of API keys to acess LLM's from, but I haven't explored this much



* The data this is built on lives in the Databricks Unity Catalog
* Google Cloud is a good alternative to databricks, as it has API keys for LLM and data storage features, but both are firewalled right now



* Refer to file directory for an explanation of the libraries in this repository



* One additional tool I used: to give Gemini web the propper context about the entire repository, it's not token efficient to upload each python file
* use Repomix to "pack" (aka convert to a more token-efficient filetype) the code into one file, then upload that file.
* you can do that here: https://repomix.com/?repo=https%3A%2F%2Fgithub.com%2Fbrian-schleisnger%2Ffin-mtkg-data-v2.git\&format=markdown\&ignore=whls%2F\*.\*%2C\*.tiktoken



