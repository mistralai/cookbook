# Samplers

When an LLM is making a prediction, it's not actually simply picking a token. It first assigns each token a degree of probability before picking among these. The question often relies on "how to choose from" those thousands of tokens. The most intuitive solution would be to always pick the best possible token, however, not all use cases would benefit from this, as sometimes you want a bit more randomness or to avoid repetition. This is where sampling takes an important role. Sampling is the process of choosing the token to be outputted among the thousands of possible ones.

A nice playground to experiment with and visualize different sampling settings: [artefact2](https://artefact2.github.io/llm-sampling/index.xhtml)

## Different Samplers Settings

<table style="border-collapse: collapse; width: 100%; text-align: center;">
  <tr>
    <td style="font-weight:bold"><a href="temperature.md">Temperature</a></td>
    <td>
    Temperature controls the overall randomness of the model's predictions. If you use a low value, the relative probability differences between the tokens become larger, making the model more deterministic. The opposite will make the model less deterministic and more creative.
    </td>
  </tr>
  <tr>
    <td style="font-weight:bold"><a href="top_p.md">Top P</a></td>
    <td>
    Top P adds up the probabilities of the topmost tokens until hitting a target percentage and will only take into consideration those tokens to pick from.
    </td>
  </tr>
  <tr>
    <td style="font-weight:bold"><a href="top_k.md">Top K</a></td>
    <td>
    Top K sets a limit on the number of top tokens that can be selected to pick from.
    </td>
  </tr>
  <tr>
    <td style="font-weight:bold">Min P</td>
    <td>
    Min P sets a minimum percentage requirement to consider tokens relative to the top token with the largest token probability.
    </td>
  </tr>
  <tr>
    <td style="font-weight:bold">Repetition Penalty</td>
    <td>
    Repetition Penalty applies a very small negative bias to all tokens that have appeared in the past to avoid repetition.
    </td>
  </tr>
</table>

> ⚠️ You can stack samplers one after the other. For example, you can use Top K to gather the top tokens and then apply Top P, or vice versa. The order of each sampler will impact how the final distribution looks.
