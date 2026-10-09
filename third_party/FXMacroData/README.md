# FXMacroData

[FXMacroData](https://fxmacrodata.com/?utm_source=github&utm_medium=referral&utm_campaign=mistral-cookbook&utm_content=readme) serves official macroeconomic releases (CPI, payrolls, GDP, policy rates, and more) through a REST API, with the publication time of each figure and a calendar of upcoming releases. USD data works without an API key.

| Notebook | Description |
|---|---|
| [fxmacrodata_macro_research_agent.ipynb](fxmacrodata_macro_research_agent.ipynb) | Give a Mistral model three function-calling tools for indicator history, the release calendar, and the indicator catalogue, then run a tool loop to answer questions such as "How has US inflation trended, and when is the next CPI release?" |
