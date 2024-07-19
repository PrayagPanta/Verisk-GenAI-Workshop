**Change Type of Runtime to use GPU.**
```
1)Install required Dependencies**
!pip install -i https://pypi.org/simple/ bitsandbytes==0.43.1
!pip install accelerate==0.32.1
!pip install transformers==4.42.3
!pip install torch==2.3.1
!pip install langchain==0.2.6
!pip install huggingface_hub==0.23.4
!pip install langchain_huggingface==0.0.3
```

```
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**2) Import and dowload model:**

```
import transformers
pipeline = transformers.pipeline(
    "text-generation",
    model= "microsoft/Phi-3-mini-4k-instruct",
    # The quantization line
    model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True},
    max_new_tokens=1024  # Increase the max tokens to your desired value
)
```

**3)Create Pipeline**

```
from langchain_huggingface.llms import HuggingFacePipeline
hf = HuggingFacePipeline(pipeline=pipeline)
```

**4) Invoke model**
```
from langchain_core.prompts import PromptTemplate

template = """<|user|>
Question: {question}<|end|>
<|assistant|>"""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is GenAI?"

print(chain.invoke({"question": question}))
```

**Effective Prompting Strategies:**
```
question = "Hi there, I need something really funny, like a joke or a funny story, but make sure it's about animals because farm animals are hilarious. Thank you so much!"
print(chain.invoke({"question": question}))


question = "Explain gravity as if you are a pirate. Please restrict the output to about 100 words."
print(chain.invoke({"question": question}))



question = """Quick advice needed:

• Issue: Forgot anniversary
• Options: Flowers, jewelry, trip
• Best choice: ?

Help!"""
print(chain.invoke({"question": question}))



question = "Workout advice"
print(chain.invoke({"question": question}))



question = '''Generate a funny joke. Here is an example:
Example: Q: Why don't scientists trust atoms? A: Because they make up everything!
Now, generate another funny joke.'''

print(chain.invoke({"question": question}))
```


```
prompt = """
Go through the below examples and generate response for the prompt at the end in similar manner.:
**Prompt:**
“Trying to impress your date with a romantic dinner? Here are a few key ingredients:

1. Candlelight ambiance.
2. Homemade pasta (bonus points if you make it yourself!).
3. A playlist that says "I'm suave but not trying too hard."
```

**Generated Output:**
Alright, here's the recipe for a perfect date night: First, dim the lights until you can barely see your own cooking skills. Then, whip up some noodles like you're auditioning for an Italian grandma role. And lastly, cue up the tunes that scream "I'm smooth like butter but won't slip off your plate." Now go knock 'em dead (figuratively)!

**Prompt:**
“Planning a cozy movie night at home? Here are the essentials:

1. A comfy blanket fort.
2. A selection of classic movies.
3. Popcorn with a dash of creativity.

**Generated Output:**
Get ready for the ultimate movie night experience! First, channel your inner architect and build a blanket fort that's as snug as a bug in a rug. Next, line up those timeless films that everyone loves. And finally, don’t just make popcorn – sprinkle some magic with your favorite seasonings. Now, sit back and let the movie magic begin!”

**Prompt:**
"Organizing a fun game night with friends? Here's what you'll need:"

<|user|>Please create a list for this prompt as well <|user|>

**Generated Output:**
<|user|>Please generate output in similar manner to above prompts<|user|>
"""

print(chain.invoke({"question": prompt}))

