# Tokenization            

For the ground truth, please refer to [mistral-common](https://github.com/mistralai/mistral-common).

## Templates

| Tokenizer               | Based on                 | Basic Chat Template                                                 | Encoding Code Sample                                                 |
|-------------------------|--------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------|
| Tokenizer V1            | `sentencepiece`          | `<s> [INST] Hello, how are you? [/INST] Fine, and you?</s> [INST] I'm doing great! [/INST] Glad to hear!</s>`          | `BOS_ID + encode("[INST] Hello, how are you? [/INST]") + encode("Fine, and you?") + EOS_ID + encode("[INST] I'm doing great! [/INST]") + encode("Glad to hear!") + EOS_ID` |
| Tokenizer V2/V3         | `sentencepiece`          | `<s>[INST] Hello, how are you?[/INST] Fine, and you?</s>[INST] I'm doing great![/INST] Glad to hear!</s>`            | `BOS_ID + INST_ID + encode("Hello, how are you?") + /INST_ID + encode("Fine, and you?") + EOS_ID + INST_ID + encode("[INST] I'm doing great! [/INST]") + /INST_ID + encode("Glad to hear!") + EOS_ID` |
| Tokenizer V3-Tekken     | `tiktoken`               | `<s>[INST]Hello, how are you?[/INST]Fine, and you?</s>[INST]I'm doing great![/INST]Glad to hear!</s>`            | `BOS_ID + INST_ID + encode("Hello, how are you?") + /INST_ID + encode("Fine, and you?") + EOS_ID + INST_ID + encode("[INST] I'm doing great! [/INST]") + /INST_ID + encode("Glad to hear!") + EOS_ID` |
