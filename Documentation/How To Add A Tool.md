So you want to add a tool.



Each tool has a few main things associated with it: the actual function, the library(ies) associated with it, the schema for the tool, the tool-schema linking, and the tool category.

* the actual function is fairly understandable - the code that actually does what the tool does. It should be easy enough to download the packed repository (see tech stacks), add that into a gemini prompt, and ask it to write you the function code. it's good practice to define the inputs and outputs of the function. "I want this to take in either a table or dataframe from memory, do \[thing] to it, and then return the average of \[column]." It's also never a bad idea to split the gemini prompting into 2. "Make a plan for how you'll implement this new tool", then another prompt "execute your plan." Also, I'd recommend figuring out what file you want the tool function to be in - at time of writing we have 3 (analytics, transformations, and visuals), but as more and more tools get added and these files get super long they might be split into more files. It's mostly a judgement call where thigns go, but I'd say for some sake of consistency keep all tools in the same category in the same file.
* the schema is the thing that the llm reads when it's deciding what tool to use and how to write the arguments. this should also be pretty simple to create - gemini should be able to do this in the same prompt as making the tool function, if it's given the schemas.py context. As of time of writing, there's one agent/schemas.py file thats getting very long, where all the schemas live. if it's decided that this gets split up later, just make sure again to keep it organized: tools in the same category are closest to each other, tools in the same toolkit file are close to each other.
* tool-schema linking is done in toolkit/\_\_init\_\_.py. you have to tell the code/llm what schema relates to what tool (even though they have the same name). When you look in this file, it's pretty much just following the same structure that exists now: import the schema, import the tool, and link them together in TOOL\_DISPATCHER.
* the tool category was created in an effort for the tool selection to be a bit more reliable (in alpha days i would ask for a regression and it would give me a scatterplot but no regression results). by splitting up tools into categories, the llm is going to be much better at deciding which tool to use. it will also split up the work: instead of one call going through 20-odd tools, one call goes through a handful of categories and the second call has just a few tools to pick from. again, it's kind of a judgment call what tool goes in what category, but gemini if given the context of the other tools could decide for you. just add the name of the tool and a short description to agen/categories.py CATEGORY\_REGISTRY in the existing format, and then loop.py will understand what category has what tools. NOTE: if a new category gets added, you have to add it to the list in agent/schemas.py class SubQuestion.



So in summary:

1. add the tool function to one of the files in toolkit
2. add the tool schema to agent/schemas.py
3. import the tool function and schemas to toolkit/\_\_init\_\_.py
4. add the tool name/description to agent/categories.py

