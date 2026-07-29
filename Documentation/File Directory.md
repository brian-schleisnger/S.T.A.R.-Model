###### **agent/**

&#x20; **\_\_init\_\_.py**

&#x09;this file contains nothing. \_\_init\_\_ files for python, however, should be in every folder. that way, other files see a package of code when going into that file, and it makes referencing a lot easier to understand.

&#x20; **cache.py**

&#x09;this is the primary long form memory system of the model. anytime a query is run, loop.py accesses cache.py at the beginning and at the end. once the question has ran and a response has been generated, loop then sends the question and response to cache. there, cache takes the response and stores the values as vectors (long strings of numbers). then, whenever a new question is run, loop goes and checks against cache's database. the new question is also embedded as a vector, and then that vector compares to all the other vectors in the cache. if there is one that is 92% similar (technically, if cosine similarity between the vectors is >.92) then the cache returns that similar response to loop, and loop skips the rest of it's steps and outputs that previous response. Note: the cache clears anytime you redeploy the app, meaning it's pretty temporary now, but as the model becomes more finalized, the cache will have longer windows to build up and be useful.

&#x20; **categories.py**

&#x09;This is a simple file that contains the map of all the tools that the model has access to. it used to be that you had to update loop.py and schemas.py, but they've been consolidated here.

&#x20; **context.py**

&#x09;this is a pretty simple file supporting the memory function of the model. it helps store the data retrieved by the llm's calls so that it can be referenced by the model in later questions in this chat string.

&#x20; **loop.py**

&#x09;This is the main file you want to reference if you want to understand the llm functionality. the main function, run\_agent\_loop, is the model's llm loop from start to finish. it: 

(1) sends the question to cache to check if there was a similar question that was asked; 

(2) filter's the schema, A.K.A. looks at the question and decides what tables should be referenced (this is before the other questions to save tokens); 

(3) runs decompose question, a prompt to break one question into multiple as necessary (ex filtering a table is one question, running a regression is a second question); 

(4) for each of the questions decompose\_question came up with, it selects a tool to use and writes the arguments to go into the tool (column names, table names, etc.); 

(5) takes the arguments written by the llm and interfaces with the relevant toolkit file (analytics, transformations, or visuals) to execute and return the tool's output; 

(6) collects the output from the tool usage and uses an llm to contextualize it; 

(7) returns all of the data, test, visuals, and tracking metrics to app.py for display.

&#x20; **memory.py**

&#x09;This file primarily serves to support inter-turn memory. it uses a package called llmlingua, which essentially takes the tokens of a message and reduces it by about 40%. it also stores a class for supporting dataframes that an llm query creates.

&#x20; **schemas.py**

&#x09;So one issue I was having constantly at the beginning was that the llms weren't very good at understanding how to write their outputs in the right way. for example, when it wanted to use the regression tool, it would pass two column names into the function, but not know that it also needed to send the table name. Schemas was created to solve that issue. for every structured output an llm needs to give in this entire pipeline, there is a schema class that enforces the specific structure and data type the llm needs to return when referencing that thing. it's built off a library called pydantic. anytime you add a new tool, you have to make a new schema for it as well.



###### **dictionaries/**

&#x09;So the llm aspects of this project are completely separate from the data tables - the only data that the llm can see is the things returned by the tools. because of this, when the llm decides the approach it's going to take for a question, it needs guidance on what the tables are and what the data looks like inside. that's what these files do: they report what the pupose of each table is, what columns there are, what's in the columns, and more things like that. if you ever decide to access more data and add it to the platform, you have to add a data dictionary as well.

&#x20; **acquisition\_data\_dictionary.json**

&#x20; **dish\_pl\_dictionary.json**

&#x20; **marketing\_spend\_dictionary.json**

&#x20; **sales\_dictionary.json**

&#x20; **subscriber\_count\_dictionary.json**



###### **Documentation/**

&#x20; **Data overview.txt**

&#x20; **File Directory.md**

&#x20; **Goals.txt**

&#x20; **Idealized Final Product.txt**

&#x20; **Intern Project Scope.txt**

&#x20; **Tech stack specifics.txt**



###### **toolkit/**

&#x20; **\_\_init\_\_.py**

&#x09;this is a helper file. for one, it helps connect the files together as described above for agent/\_\_init\_\_.py. this one isn't empty: it has a couple of functions and a bunch of imports. these imports and connections serve to link the tools with the schemas related to them for the llm to understand what is what.

&#x20; **analytics.py**

&#x09;this is the workhorse file with all of the actual statistical/data mining stuff. if you want the model to do something it can't right now, chances are you'll put it in here. we have the following functions.

* run ols regression is linear regression, probably the most used tool outside of sql querying.
* run forecasting tool is pretty much waht it sounds like, takes in some historical data and tries to predict what will that data will look like moving forward.
* random forest tool is a machine learning tool that allows for non-linear regression or classification; it takes in a bunch of data, samples that data a bunch of times, generates a decision tree for each sample that uses some variables to predicat the target variable, and then "averages" out all those decision trees to answer the question "how much do these variables impact this variable really?".
* pca tool, short for principal component analysis, takes in a bunch of columns and basically simplifies all the columns down to just a few derived columns that don't correlate with each other, it'll return something like "95% of variation in the data is explained by this factor".
* kmeans clustering takes in abunch of data points and tries to divide them up in the best way possible; it could be useful for identifying data-driven market segments.
* scenario planning tool is the what-if function: takes in a variable that changes, a variable that stays constant, and a variable that you want to measure; if we reduce spend by 10% how will activations be affected?.
* neural network tool is another machine learning method for non-linear relationships; it takes a long time to run, but is probably the most industry-standard for gaining insights.
* run optimization is the standard linear programming optimization method; it takes in equations to maximize or minimize, as well as bounds on certain variables, and adjusts the inputs of the function to get to the best possible value.
* mutual information tool is designed to tell you how much "information" one variable give you about another variable - likely won't run, more for things like feature selection if the ml elements ever get heavier

&#x20; **base.py**

&#x09;this is basically a file designed to house a bunch of random stuff. it's where the model endpoints and selection functions live. it has token tracking functions and the cost function to determine the conversion from tokens to $. it houses the authentication token it uses to access databricks. it handles the llm client: basically wrapping the endpoint in a class for standardization purposes. it has table linking stuff; what columns go together, and what dictionaries go with what. finally, it has the actual raw llm call function that loop accesses, and the direct sql call function that interacts with databricks.

&#x20; **Transformations.py**

&#x09;originally, this and analytics.py were all the same function, but I split them up to ensure not file as too long. this is where a few things live: the sql and python code tools, the data transformation tools that link and pivot dataframes, ratio functions (and the custom built cpa function), and the link tables function (which supports queries from multiple databricks tables

&#x20; **Validators.py**

&#x09;This was a file i recently created to abstract out a few things and make sure no file got to long. it has the main purpose of ensuring that anything the llm writes is in the right format, as well as safe to run (aka won't delete rows of data or something along those lines).

&#x20; **visuals.py**

&#x09;this is the counterpart to analytics.py. it serves teh same function, it just houses the visual tools so that analytics isn't like 2000 lines. there are 4 visuals I gave it: scatterplots, bar charts, histograms, and line charts. it also has a few helper functions to grab the right columns and link tables and columns together for visualization purposes.



###### **whls/**

&#x09;So python is a very collaborative coding language. instead of having every line of code used be written by me, I can use what are called libraries. I'll also sometimes call them packages because the terms are sort of interchangeable. If you're working solely on your computer, to get these libraries, you run a command in your PowerShell 'pip install \[library name]'. what this command does is it goes to pypi.org (the website maintained by official python people that houses all the library files), finds the library you are looking for, and adds the code to your computer. the files it fetches have the extension .whl, and are commonly called wheels, but you can sort of think of them as zip files full of code. once you pip install something, you can then reference that library in any code you run on your computer.

&#x09;So the code for the app isn't running on your computer, though, it's running through databricks when you boot it up. so it needs to access the libraries somehow. typically, it would do this through a file labelled requirements.txt, with a list of all the libraries you need, and then the app goes to pypi.org to find all the relevant .whl files. well because of firewall restrictions (either by databricks admins or broader internet restrictions, i'm not sure), the app is blocked from going to pypi.org. However, you can get around this by manually downloading the .whl files yourself, then using requirements.txt to point to those local wheel files. that's what I've decided to do.

&#x09;if you are updating the code, and you need to use a new package, here's what you do. (1) go to pypi.org, search up the package, and download the relevant wheel file. there will be a bunch of options - pick the one with either none-any in the file name, or cp11, manylinux, and x86 in the file name. (2) google '\[library name] package dependencies'. this should give you a readout of the other libraries the one you're trying to install is built on top of. repeat step 1 for each of these libraries. (3) move all the wheel files you downloaded into this folder. (4) go to requirements.txt and add all the file names here.

&#x09;below I'm going to list the important libraries we're using and their purpose. some of them might not have a wheel files in this folder - that's because they come default installed to databricks.

* databricks.sdk is the way we access the things from databricks we use, like the model endpoints and the authentication stuff.
* instructor is a library that works with pydantic to structure llm outputs so that functions can read them and identify what they need. l
* lmlingua is a library that supports the compression of tokens for better cost management, along with tiktoken.
* mlflow is how we track our token usage across all the llm calls. numpy is a standard library that gives up better ways to do math and store values.
* openai is a wrapper for our llms; basically it makes everything more standardized and easier to move off databricks if necessary.
* pandas is the most common library; it's used to store data in structure tables called dataframes.
* pg8000 is a helper library that supports our ability to connect to sql tables and run sql queries.
* plotly is a visual library that manages all of our graphs made by the tools.
* pydantic works with instructor to standardize and control the outputs of the llm, so it interfaces better with the code.
* scikit-learn, also called sklearn, is a library with a bunch of data analysis and machine learning functions prebuilt; it servres as the baseline for a bunch of the analysis.py functions.
* sqlalchemy works with pg8000 to manage how we write and run sql queries throughout the model.
* statsmodels is a dedicated package for regression analysis.
* streamlit is a user interface library - it works to display all the thing syou see on the webpage.
* tiktoken works with llmlingua for token compression and memory and things like that.



**.gitattributes**

&#x09;this is a technical thing for one of the libraries, torch, that's really big. see app.py for more details.

**.gitignore**

&#x09;this it another technical file. basically any file that's in the repository and in this list is ignored every time you commit and push to the git. I don't think this is totally necessary, but multiple ai stuff I've used to help make the program have said I need it so ok.

**app.py**

&#x09;so when databricks deploys the app. it reaches into the git and downloads the repository I've told it to point to. once it does that, it looks for a file called app.py to run. moral of the story is there always has to be a file called app.py in the main folder. in the file, I've put two things: all the user interface code and a function called bootstrap environment.

&#x09;so there's one library we use, i don't remember which one, that is dependent on a package called torch. torch is a standard llm package and can be used for a lot more than we're using it for. but one thing to know is it's really big - i think around 150 mbs. that mean's it's bigger than the maximum file size git allows. the workaround I've created is I basically split the .whl file for torch in two parts. because of this, anytime the app boots up for the first time on a device, it needs to smash those two parts together. that's what bootstrap\_enviroment does.

&#x09;the rest of the file, and the main purpose of app.py, is manage all of the user interface (ui). this is probably the clearest commented file - just look at each esection, and it should have a description  of what that part is doing. anything technical, like running the model loop, is done via another file.

**app.yaml**

&#x09;this is another necessary file that databricks looks for upon deployment. it holds some specific host details, but in general don't touch this unless there's specifically an issue with it.

**logo1.png**

&#x09;this is used by app.py in the ui stuff so that the dish logo is the little icon on the tab whn you boot it up.

**requirements.txt**

&#x09;so when databricks deploys the app, it looks to app.py. when it sees that app.py has libraries it references, it then looks to requirements.txt, which tells databricks where to look for said package. that's why we need requirements.txt - it points databricks to where the libraries can be found. in most cases, it's inthe whls/ folder. in others, it points to the databricks environment for all the standard libraries that come pre-installed.

**style.css**

&#x09;this is a formatting file. app.py references it when it is formatting. I've never worked with .css before, so if you have questions about it just have gemini make it for you.

