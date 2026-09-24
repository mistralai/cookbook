# Building LLM Agents in C++ with agt and Mistral

**Author:** Pierre Evenou
**GitHub:** [@0x9dhcf](https://github.com/0x9dhcf)

## Overview

This cookbook shows how to use Mistral models from C++ to build agentic
applications,from a simple completion loop to tool calling, streaming with
lifecycle hooks, and MCP server integration.

All examples use [agt](https://github.com/0x9dhcf/agt), a minimal C++
library for building LLM-powered agents. agt supports tool calling,
streaming, MCP integration, multi-agent composition, and persistent session
management. It is provider-agnostic but pairs well with Mistral thanks to
native support for Mistral-specific features like
[thinking effort](https://docs.mistral.ai/capabilities/reasoning/#thinking-effort)
control.

This is an independent open-source library, shared as an example and a
starting point. It is not a production-ready framework and does not aim to
replace more complete solutions. Think of it as a C++ take on the kind of
agentic loop you would find in a typical agents SDK useful for developers
working in C++ codebases, exploring embedded agents, or simply curious about
how these patterns translate outside of Python.

agt grew out of [chatty](https://github.com/0x9dhcf/chatty), a personal
terminal assistant built for daily use, which needed a lightweight agentic
loop with no heavy runtime dependencies.

Contributions and feedback welcome.

## Requirements

- C++23 compiler (GCC 13+ or Clang 17+)
- CMake 3.21+
- libcurl
- SQLite3
- A Mistral API key from [console.mistral.ai](https://console.mistral.ai/api-keys/)

## Getting CPM

All examples use [CPM](https://github.com/cpm-cmake/CPM.cmake) to fetch agt
automatically. Download it once into a `cmake/` folder:

```sh
mkdir cmake
curl -Lo cmake/CPM.cmake \
  https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
```

---

## Example 1: Direct completion

The simplest possible use of agt: a multi-turn conversation loop with no
tools. The `thinking_effort` parameter is set to `"low"`, which controls how
much reasoning the Mistral model performs before answering. Valid values are
`"low"`, `"medium"`, and `"high"` — lower effort gives faster, more concise
replies while higher effort enables deeper reasoning on complex questions.

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.21)
project(complete CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/CPM.cmake)
CPMAddPackage("gh:0x9dhcf/agt#main")

add_executable(complete main.cpp)
target_link_libraries(complete PRIVATE agt::agt)
```

### main.cpp

```cpp
#include <agt/llm.hpp>
#include <iostream>
#include <print>

int main() {
  const std::string model = "mistral-small-latest";
  const std::string thinking = "low";

  agt::Llm llm(agt::Provider::mistral, model, getenv("MISTRAL_API_KEY"));

  std::println("provider : mistral");
  std::println("model    : {}", model);
  std::println("thinking : {}", thinking);
  std::println("---");

  agt::Json req = {{"system", "You are a helpful assistant."},
                   {"messages", agt::Json::array()}};

  std::string line;
  std::print(">>> ");
  while (std::getline(std::cin, line)) {
    req["messages"].push_back({{"role", "user"}, {"content", line}, {"thinking_effort", thinking}});

    auto res = llm.complete(req);
    auto content = res.value("content", "");
    std::println("{}\n", content);

    req["messages"].push_back({{"role", "assistant"}, {"content", content}});
    std::print(">>> ");
  }
}
```

### Build and run

```sh
cmake -B build
cmake --build build

export MISTRAL_API_KEY=...
./build/complete
```

### Expected output

```
provider : mistral
model    : mistral-small-latest
thinking : low
---
>>> What is 2+2?
2 + 2 equals **4**.

>>>
```

---

## Example 2: Agent with tools

Add tool calling to the loop. Mistral's function calling lets the model
decide when to invoke tools and incorporate their results into the response.
agt handles the dispatch automatically. Conversation history is persisted
across runs using a SQLite-backed session.

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.21)
project(agent CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/CPM.cmake)
CPMAddPackage("gh:0x9dhcf/agt#main")

add_executable(agent main.cpp)
target_link_libraries(agent PRIVATE agt::agt)
```

### main.cpp

```cpp
#include <agt/agent.hpp>
#include <agt/llm.hpp>
#include <agt/runner.hpp>
#include <agt/session.hpp>
#include <agt/tool.hpp>
#include <print>

class get_weather : public agt::Tool {
public:
  const char* name() const noexcept override { return "get_weather"; }
  const char* description() const noexcept override {
    return "Get current weather for a city.";
  }

  agt::Json parameters() const override {
    return {{"type", "object"},
            {"properties", {{"city", {{"type", "string"}}}}},
            {"required", {"city"}}};
  }

  agt::Json execute(const agt::Json& input, void*) override {
    return {{"temperature", 22}, {"city", input["city"]}};
  }
};

int main() {
  agt::Llm llm(agt::Provider::mistral,
               "mistral-small-latest",
               getenv("MISTRAL_API_KEY"));

  agt::Agent agent;
  agent.instructions = "You are a helpful assistant.";
  agent.tools        = {std::make_shared<get_weather>()};
  agent.session      = agt::make_sqlite_session("history.db", "session-1");

  agt::Runner runner;
  auto resp = runner.run(llm, agent, "What is the weather in Paris?");
  std::println("{}", resp.content);
}
```

### Build and run

```sh
cmake -B build
cmake --build build

export MISTRAL_API_KEY=...
./build/agent
```

### Expected output

```
The current temperature in Paris is **22°C**.
```

The model called `get_weather` with `{"city": "Paris"}` behind the scenes,
received the tool result, and formulated the final answer.

---

## Example 3: Streaming with lifecycle hooks

A multi-tool travel advisor that uses Mistral's streaming API to deliver
tokens as they arrive and logs the full agentic lifecycle — LLM calls, tool
invocations, and token counts — to stderr. Mistral supports parallel tool
calls: the model can request multiple tools in a single turn, which agt
dispatches concurrently.

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.21)
project(streaming CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/CPM.cmake)
CPMAddPackage("gh:0x9dhcf/agt#main")

add_executable(streaming main.cpp)
target_link_libraries(streaming PRIVATE agt::agt)
```

### main.cpp

```cpp
#include <agt/agent.hpp>
#include <agt/llm.hpp>
#include <agt/runner.hpp>
#include <agt/session.hpp>
#include <agt/tool.hpp>
#include <print>

class get_weather : public agt::Tool {
public:
  const char* name() const noexcept override { return "get_weather"; }
  const char* description() const noexcept override {
    return "Get current weather for a city including temperature, humidity and "
           "conditions.";
  }
  agt::Json parameters() const override {
    return {{"type", "object"},
            {"properties", {{"city", {{"type", "string"}}}}},
            {"required", {"city"}}};
  }
  agt::Json execute(const agt::Json& input, void*) override {
    auto city = input["city"].get<std::string>();
    if (city == "Paris")
      return {{"city", city}, {"temp_c", 18}, {"humidity", 72},
              {"conditions", "partly cloudy"}, {"wind_kmh", 14}};
    if (city == "Tokyo")
      return {{"city", city}, {"temp_c", 26}, {"humidity", 85},
              {"conditions", "humid and sunny"}, {"wind_kmh", 8}};
    if (city == "New York")
      return {{"city", city}, {"temp_c", 12}, {"humidity", 55},
              {"conditions", "clear skies"}, {"wind_kmh", 22}};
    return {{"city", city}, {"temp_c", 20}, {"humidity", 60},
            {"conditions", "fair"}, {"wind_kmh", 10}};
  }
};

class get_events : public agt::Tool {
public:
  const char* name() const noexcept override { return "get_events"; }
  const char* description() const noexcept override {
    return "List upcoming cultural events and activities in a city.";
  }
  agt::Json parameters() const override {
    return {{"type", "object"},
            {"properties", {{"city", {{"type", "string"}}}}},
            {"required", {"city"}}};
  }
  agt::Json execute(const agt::Json& input, void*) override {
    auto city = input["city"].get<std::string>();
    if (city == "Paris")
      return {{"city", city},
              {"events", {{{"name", "Nuit des Musées"},
                           {"date", "May 17"},
                           {"description", "Free entry to museums across Paris"}},
                          {{"name", "Roland-Garros"},
                           {"date", "May 25 - Jun 8"},
                           {"description", "French Open tennis tournament"}},
                          {{"name", "Fête de la Musique"},
                           {"date", "Jun 21"},
                           {"description", "City-wide free music festival"}}}}};
    if (city == "Tokyo")
      return {{"city", city},
              {"events", {{{"name", "Sanja Matsuri"},
                           {"date", "May 16-18"},
                           {"description", "Traditional festival at Asakusa Shrine"}},
                          {{"name", "Tokyo Rainbow Pride"},
                           {"date", "Apr 19-20"},
                           {"description", "Parade and events in Yoyogi Park"}}}}};
    return {{"city", city}, {"events", agt::Json::array()}};
  }
};

int main() {
  agt::Llm llm(agt::Provider::mistral,
               "mistral-small-latest",
               getenv("MISTRAL_API_KEY"));

  agt::Agent agent;
  agent.instructions =
      "You are a travel advisor. When comparing cities, call tools for each "
      "city, then write a detailed comparison covering weather, events and "
      "your recommendation. Use markdown formatting.";
  agent.tools = {std::make_shared<get_weather>(),
                 std::make_shared<get_events>()};

  agt::RunnerHooks hooks;

  hooks.on_start = [] {
    std::println(stderr, "[stream] run started");
  };

  hooks.on_llm_start = [](const agt::Llm&, const agt::Json&) {
    std::println(stderr, "[stream] calling LLM ...");
  };

  hooks.on_tool_start = [](const agt::Tool& t, const agt::Json& args) {
    std::println(stderr, "[stream] tool {}({})", t.name(), args.dump());
    return true; // allow execution
  };

  hooks.on_tool_stop = [](const agt::Tool& t, const agt::Json&,
                          const agt::Json& result) {
    std::println(stderr, "[stream] tool {} -> {}", t.name(), result.dump());
  };

  hooks.on_token = [](const std::string& t) {
    std::print("{}", t);
  };

  hooks.on_stop = [] {
    std::println(stderr, "\n[stream] run finished");
  };

  agt::Runner runner;
  auto res = runner.run(
      llm, agent,
      "I'm planning a spring trip and hesitating between Paris and Tokyo. "
      "Compare the weather and upcoming events for both cities, then give "
      "me your recommendation.",
      {}, hooks);

  std::println("");
  std::println(stderr, "[stream] tokens in={} out={}", res.input_tokens,
               res.output_tokens);
}
```

### Build and run

```sh
cmake -B build
cmake --build build

export MISTRAL_API_KEY=...
./build/streaming
```

### Expected output

Stderr shows the agentic lifecycle — note how the model calls all four tools
in parallel before composing its final answer:

```
[stream] run started
[stream] calling LLM ...
[stream] tool get_weather({"city":"Paris"})
[stream] tool get_weather({"city":"Tokyo"})
[stream] tool get_events({"city":"Paris"})
[stream] tool get_events({"city":"Tokyo"})
[stream] tool get_weather -> {"city":"Paris","conditions":"partly cloudy","humidity":72,"temp_c":18,"wind_kmh":14}
[stream] tool get_weather -> {"city":"Tokyo","conditions":"humid and sunny","humidity":85,"temp_c":26,"wind_kmh":8}
[stream] tool get_events -> {"city":"Paris","events":[{"date":"May 17","description":"Free entry to museums across Paris","name":"Nuit des Musées"},{"date":"May 25 - Jun 8","description":"French Open tennis tournament","name":"Roland-Garros"},{"date":"Jun 21","description":"City-wide free music festival","name":"Fête de la Musique"}]}
[stream] tool get_events -> {"city":"Tokyo","events":[{"date":"May 16-18","description":"Traditional festival at Asakusa Shrine","name":"Sanja Matsuri"},{"date":"Apr 19-20","description":"Parade and events in Yoyogi Park","name":"Tokyo Rainbow Pride"}]}
[stream] calling LLM ...
...
streamed markdown comparison and recommendation...
...
[stream] run finished
Would you like recommendations on specific activities or itineraries for either city?
[stream] tokens in=675 out=659
```

---

## Example 4: MCP server integration

Connect to an MCP server to discover and use tools dynamically. The Mistral
model sees the MCP-provided tools exactly like native ones and calls them
through the same function calling mechanism. This example uses the filesystem
MCP server. Requires Node.js and npx.

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.21)
project(mcp CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/CPM.cmake)
CPMAddPackage("gh:0x9dhcf/agt#main")

add_executable(mcp main.cpp)
target_link_libraries(mcp PRIVATE agt::agt)
```

### main.cpp

```cpp
#include <agt/agent.hpp>
#include <agt/llm.hpp>
#include <agt/mcp.hpp>
#include <agt/runner.hpp>
#include <print>

int main() {
  agt::Llm llm(agt::Provider::mistral,
               "mistral-small-latest",
               getenv("MISTRAL_API_KEY"));

  agt::McpServer server({
    .transport = agt::McpTransport::stdio,
    .name      = "filesystem",
    .command   = "npx",
    .args      = {"-y", "@modelcontextprotocol/server-filesystem", "/tmp"},
  });
  server.connect();

  agt::Agent agent;
  agent.instructions = "You are a helpful assistant.";
  agent.tools        = server.tools();

  agt::Runner runner;
  auto resp = runner.run(llm, agent, "List the files in /tmp.");
  std::println("{}", resp.content);
}
```

### Build and run

```sh
cmake -B build
cmake --build build

export MISTRAL_API_KEY=...
./build/mcp
```

### Expected output

```
Here is the list of files and directories in `/tmp`:
...
```

The model used the MCP filesystem server's `list_directory` tool to read
`/tmp` and formatted the result. The actual output will vary depending on
your system.

---

## Library

agt is in active early development. The current foundation covers
multi-provider LLM calls, tool calling, MCP, sessions, streaming, and
lifecycle hooks. See
[ROADMAP.md](https://github.com/0x9dhcf/agt/blob/main/ROADMAP.md) for
upcoming features.

Source: [github.com/0x9dhcf/agt](https://github.com/0x9dhcf/agt) — MIT license
