# Top K

**Top K** is a simpler sampler setting similar to Top P that enables delimiting how many tokens will be considered not by using a percentage but a number of tokens. The goal is to set a hard limit of tokens that can be considered.

If Top K is 5, then only the top 5 tokens will be considered.

## Visualization
For the following demonstrations, we will be setting the Temperature first and then a Top K of 3. Note that a Temperature of 0 will always be deterministic, and in this scenario, Top K won't change anything.
The order of events is as follows:
- First, the Temperature is applied.
- After that, the Top K of 3 keeps only the top 3 tokens.
- Their probabilities change due to the other tokens no longer being an option.

The distribution would change as follows for the following question: *"What is the best mythical creature? Answer with a single word."*

<table style="border-collapse: collapse; width: 100%; text-align: center;" align="center">
  <tr>
    <td style="white-space: nowrap;">Tokens</td>
    <td style="white-space: nowrap;">Dragon</td>
    <td style="white-space: nowrap;">Un</td>
    <td style="white-space: nowrap;">Drag</td>
    <td style="white-space: nowrap;">Peg</td>
    <td style="white-space: nowrap;">Phoenix</td>
  </tr>
  <tr>
    <td>T=0</td>
    <td>100%</td>
    <td>0%</td>
    <td>0%</td>
    <td>0%</td>
    <td>0%</td>
  </tr>
  <tr>
    <td>T=0.2</td>
    <td>65.1%</td>
    <td>34.8%</td>
    <td>0%</td>
    <td>0%</td>
    <td>0%</td>
  </tr>
  <tr>
    <td>T=0.4</td>
    <td>56.6%</td>
    <td>41.4%</td>
    <td>1.5%</td>
    <td>0.2%</td>
    <td>0%</td>
  </tr>
  <tr>
    <td>T=0.8</td>
    <td>46.1%</td>
    <td>39.4%</td>
    <td>7.6%</td>
    <td>2.7%</td>
    <td>1.3%</td>
  </tr>
  <tr>
    <td>T=1.6</td>
    <td>21.3%</td>
    <td>19.7%</td>
    <td>8.6%</td>
    <td>5.2%</td>
    <td>3.6%</td>
  </tr>
</table>

  <div style="margin-right: 20px; text-align: center;" align="center">
    <sub><sup>Top 5 tokens using Mistral 7B at 4 bits precision for different Temperature values.</sup></sub>
  </div>

  <div style="margin-right: 20px; text-align: center;" align="center">
    <span style="font-size: 24px;">&darr;</span>
  </div>

<table style="border-collapse: collapse; width: 100%; text-align: center;" align="center">
  <tr>
    <td style="white-space: nowrap;">Tokens</td>
    <td style="white-space: nowrap;">Dragon</td>
    <td style="white-space: nowrap;">Un</td>
    <td style="white-space: nowrap;">Drag</td>
    <td style="white-space: nowrap;">Peg</td>
    <td style="white-space: nowrap;">Phoenix</td>
  </tr>
  <tr>
    <td>T=0</td>
    <td>100%</td>
    <td>0%</td>
    <td>0%</td>
    <td>X</td>
    <td>X</td>
  </tr>
  <tr>
    <td>T=0.2</td>
    <td>65.1%</td>
    <td>34.8%</td>
    <td>0%</td>
    <td>X</td>
    <td>X</td>
  </tr>
  <tr>
    <td>T=0.4</td>
    <td>56.9%</td>
    <td>41.6%</td>
    <td>1.5%</td>
    <td>X</td>
    <td>X</td>
  </tr>
  <tr>
    <td>T=0.8</td>
    <td>46.1%</td>
    <td>39.4%</td>
    <td>7.6%</td>
    <td>X</td>
    <td>X</td>
  </tr>
  <tr>
    <td>T=1.6</td>
    <td>21.3%</td>
    <td>19.7%</td>
    <td>8.6%</td>
    <td>X</td>
    <td>X</td>
  </tr>
</table>

  <div style="margin-right: 20px; text-align: center;" align="center">
    <sub><sup>Top K will only consider the top 3 tokens.</sup></sub>
  </div>

  <div style="margin-right: 20px; text-align: center;" align="center">
    <span style="font-size: 24px;">&darr;</span>
  </div>

<table style="border-collapse: collapse; width: 100%; text-align: center;" align="center">
  <tr>
    <td style="white-space: nowrap;">Tokens</td>
    <td style="white-space: nowrap;">Dragon</td>
    <td style="white-space: nowrap;">Un</td>
    <td style="white-space: nowrap;">Drag</td>
    <td style="white-space: nowrap;">Peg</td>
    <td style="white-space: nowrap;">Phoenix</td>
  </tr>
  <tr>
    <td>T=0</td>
    <td>100%</td>
    <td>0%</td>
    <td>0%</td>
    <td>0%</td>
    <td>0%</td>
  </tr>
  <tr>
    <td>T=0.2</td>
    <td>65.1%</td>
    <td>34.8%</td>
    <td>0%</td>
    <td>0%</td>
    <td>0%</td>
  </tr>
  <tr>
    <td>T=0.4</td>
    <td>56.6%</td>
    <td>41.4%</td>
    <td>1.5%</td>
    <td>0%</td>
    <td>0%</td>
  </tr>
  <tr>
    <td>T=0.8</td>
    <td>49.5%</td>
    <td>42.3%</td>
    <td>8.2%</td>
    <td>0%</td>
    <td>0%</td>
  </tr>
  <tr>
    <td>T=1.6</td>
    <td>42.9%</td>
    <td>39.7%</td>
    <td>17.3%</td>
    <td>0%</td>
    <td>0%</td>
  </tr>
</table>

  <div style="margin-right: 20px; text-align: center;" align="center">
    <sub><sup>The other tokens are removed, and the probabilities are updated.</sup></sub>
  </div>

Top K only keeps the top tokens regardless of probabilities, it's a more naive way of sampling compared to Top P.

# What Have We Learnt?

1. **Role of Top K**: Top K is a sampler setting that limits the number of tokens considered to a specified number. Unlike Top P, which uses a probability mass, Top K sets a hard limit on the number of tokens that can be considered. For example, if Top K is set to 5, only the top 5 tokens will be considered.

2. **Interaction with Temperature**: Similar to Top P, Top K is usually applied after the Temperature setting. The Temperature adjusts the probability distribution of the tokens, and then Top K narrows down the selection to the top K tokens.

3. **Impact on Outputs**: By using Top K, the model focuses on the most likely tokens, which can help in maintaining a certain level of quality and coherence in the outputs. However, it does so in a more straightforward manner compared to Top P, as it does not consider the cumulative probabilities.

Visit other sampler settings here -> <a href="README.md">Sampling</a>
