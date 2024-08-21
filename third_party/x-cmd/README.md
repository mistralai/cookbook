## The Mistral AI Command Line Client

The **mistral module** is a command-line client tool built by the x-cmd team using the Mistral AI API.  Written in posix shell and awk, it uses `curl` to send API requests. 

- [Installing](https://x-cmd.com/start/) x-cmd:
    ```sh
    eval "$(curl https://get.x-cmd.com)"
    # or
    eval "$(wget -O- https://get.x-cmd.com)"
    ```

## Getting started

1. Configuring `x mistral`:
    ```sh
    x mistral init
    # or
    x mistral --cfg apikey=<Mistral AI API Key>
    x mistral --cfg model=<Mistral Model>
    ```
    ![x mistral init](static/mistral.init.png)

2. Having a conversation with Mistral AI：
    ```sh
    x mistral chat request "hello"
    # The command alias is @mistral
    @mistral "hello"
    @mistral --file <file_path> "Translate to French"
    ```
    ![@mistral file](static/mistral.chat.1.png)

3. Having a conversation in an interactive mode.
    ```sh
    # The command `x jina r` uses Jina.ai to extract content from web pages
    x jina r "https://www.x-cmd.com/start/guide" | @mistral
    ```
    ![mistral repl](static/x.mistral.png)

## Command Line Options

We offer the `x mistral` and `@mistral` commands, where `x mistral` focuses on model configuration and download management, while `@mistral` emphasizes model applications. Their command-line options are as follows: 

1. `x mistral`:
    ```sh
    SUBCOMMANDS:
        init    Initialize the configuration using interactive mode
        --cur   current session default value management
        --cfg   Manage config item like apikey, etc
        chat    chat with mistral
        model   Model viewing and management
    ```
2. `@mistral`:
    ```sh
    -t,--temperature     Control the diversity of model generated results, the range is [0 ~ 1], when the temperat
    -e                   Send the variable value as context to AI
    -f,--file            Send file content as context to AI
    -n,--history         Specify the number of history as context
    -p                   Specify to get pipe content
    -P,--nopipe          Specify not to get pipe content
    -c                   Confirm before sending the request content to AI
    --jina               Through jina reader, send the webpage content as context to AI
    --ddgo               Send ddgo search content as context to AI
    --tldr               Send tldr content as context to AI
    --eval               Send the execution command results as context to AI
    --kv                 Send key-value pairs as context to AI
    --session            Specify session value
    --minion             Specify minion file
    --model              Specify AI model
    --edit               Edit the request content in the terminal
    --numbered           List the data with line numbers and send it
    --question           Request content
    ```
