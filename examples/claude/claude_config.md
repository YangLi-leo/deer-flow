# Using Claude Models with DeerFlow

DeerFlow supports Anthropic's Claude models through the langchain-anthropic integration. This example shows how to configure and use Claude models for your research tasks.

## Configuration

To use Claude models, update your `conf.yaml` file with the following configuration:

```yaml
BASIC_MODEL:
  model: "anthropic/claude-3-sonnet-20240229"
  api_key: $ANTHROPIC_API_KEY
  max_tokens: 4096
  temperature: 0.7
```

Remember to set your Anthropic API key in your environment variables or directly in the `.env` file:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Available Claude Models

DeerFlow supports the following Claude models:

- `anthropic/claude-3-opus-20240229`: Most capable model for complex tasks
- `anthropic/claude-3-sonnet-20240229`: Balanced performance and cost
- `anthropic/claude-3-haiku-20240307`: Fast and cost-effective
- `anthropic/claude-3.5-sonnet-20240620`: Latest model with improved capabilities

## Best Practices

When using Claude models:

1. Provide clear and specific research questions
2. Upload relevant documents to reduce context-setting
3. Use step-by-step reasoning in your prompts
4. Consider using Claude's data analysis capabilities for CSV data
5. Verify sources and citations in the research output

For more details on Claude's capabilities, see the [How to use Claude for Deep Research](/examples/how_to_use_claude_deep_research.md) guide.
