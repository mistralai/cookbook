# Chainlit & Mistral reasoning

This application uses the Chainlit UI framework along with Mistral's tool calls to answer complex questions requiring multiple-step reasoning 🥳

## Requirements

_Versions used for the demo are `chainlit===1.1.305` and `mistralai===0.4.1`_

We manage environment variables with `python-dotenv`.

You will need a Mistral API key, which you can get at https://console.mistral.ai/api-keys/.
Make sure to set it as `MISTRAL_API_KEY=` in a `.env` environment.

```shell
pip install chainlit mistralai
```

Optionally, you can get a Literal AI API key from [here](https://docs.getliteral.ai/get-started/installation#how-to-get-my-api-key)
and set it as `LITERAL_API_KEY` in your `.env`. This will allow you to visualize the flow of your application.

## Run the Chainlit application

The full application code lives in `app.py`. To run it, simply execute the following line:

```shell
chainlit run app.py
```

This will spin up your application on http://localhost:8080! 🎉

For more a more step-by-step instructive tutorial on writing the application code, you call follow the `Chainlit - Mistral reasoning` notebook!
