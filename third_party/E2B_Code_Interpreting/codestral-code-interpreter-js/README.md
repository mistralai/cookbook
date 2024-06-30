# AI Code Execution with Mistral's Codestral

This example uses is an AI data analyst using Mistral's Codestral as the LLM, and the [Code Interpreter SDK](https://github.com/e2b-dev/code-interpreter) by E2B for the code interpreting capabilities.

The AI agent performs a data analysis task on an uploaded CSV file, executes the AI-generated code in the sandboxed environment by E2B, and returns a chart, saving it as a PNG file.


# Installation

## 1. Install dependencies

Ensure all dependencies are installed:

```
npm install
```

## 2. Set up environment variables

Create a `.env` file in the project root directory and add your API keys:

- Copy `.env.template` to `.env`
- Get the [E2B API KEY](https://e2b.dev/docs/getting-started/api-key)
- Get the [MISTRAL API KEY](https://console.mistral.ai/api-keys/)

## 3. Run the program

```
npm run start
```

The script performs the following steps:
    
- Loads the API keys from the environment variables.
- Uploads the CSV dataset to the E2B sandboxed cloud environment.
- Sends a prompt to the Codestal model to generate Python code for analyzing the dataset.
- Executes the generated Python code using the E2B Code Interpreter SDK.
- Saves any generated visualization as PNG file.
  

After running the program, you should get the result of the data analysis task saved in a `image_1.png` file. You should see a plot like this:

![Example of the output](image_1.png)


# Connect with E2B & learn more
If you encounter any problems, please let us know at our [Discord](https://discord.com/invite/U7KEcGErtQ).

Check the [E2B documentation](https://e2b.dev/docs) to learn more about how to use the Code Interpreter SDK.