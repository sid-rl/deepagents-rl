<!-- Corrected by Docs Reviewer on 2025-10-23T10:35:37.240919 -->
<!-- Original file: /Users/sydney_runkle/oss/deepagents/docs_reviewer/example_docs.md -->

# Example Documentation

This is an example markdown file with code snippets for testing the docs reviewer.

## Basic Python Example

Here's a simple Python example:

```python
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
```

## LangChain Example

This example shows how to use LangChain:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Initialize the model
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Create a message
messages = [HumanMessage(content="What is the capital of France?")]

# Invoke the model
response = model.invoke(messages)
print(response.content)
```

## Example with Error

This example has a deliberate error:

```python
def add_numbers(a, b):
    return a + b

# This will fail - undefined variable
result = add_numbers(x, y)
print(result)
```

## JavaScript Example

Here's a JavaScript snippet:

```javascript
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

console.log(fibonacci(10));
```

## Shell Example

```bash
echo "Hello from bash"
ls -la
```

## Non-Executable Example

This is just output, not meant to be executed:

```text
Output:
Hello, World!
Success!
```
