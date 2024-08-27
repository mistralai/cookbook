# Top P

**Top P** is a sampler setting that enables delimiting how many tokens will be considered based on a percentage. The idea behind it is to not consider the entirety of the vocabulary when sampling but only a certain amount among a certain probability mass.

Let's say we set a Top P of 0.1, this would translate to only considering the best tokens whose sum of probabilities adds up to 10%.

## Visualization
For the following demonstrations, we will be setting the temperature first and then a Top P of 50%. Note that a temperature of 0 will always be deterministic, and in this scenario, Top P won't change anything.
The order of events is as follows:
- First, the temperature is applied.
- After that, the Top P of 0.5 keeps only the most likely tokens.
- Their probabilities change due to certain tokens no longer being an option.

The distribution would change as follows:

<div style="justify-content: center; align-items: center;">
  <div style="margin-right: 20px; text-align: center;" align="center">
    <img src="top_barplot.png" alt="Example Image" width="50%">

<sub><sup>Different `temperature` values and the top 5 tokens using Mistral 7B at 4 bits precision.</sup></sub>
  </div>

  <div style="margin-right: 20px; text-align: center;">
    <span style="font-size: 24px;">&darr;</span>
  </div>

  <div style="margin-right: 20px; text-align: center;">
    <img src="top_barplot_black.png" alt="Example Image" width="50%">

<sub><sup>Top P will only consider the topmost tokens until hitting the 50% target.</sup></sub>
  </div>

  <div style="margin-right: 20px; text-align: center;">
    <span style="font-size: 24px;">&darr;</span>
  </div>

  <div style="text-align: center;">
    <img src="top_barplot_final.png" alt="Example Image" width="50%">

<sub><sup>The other tokens are removed, and the probabilities are updated.</sup></sub>
  </div>
</div>
