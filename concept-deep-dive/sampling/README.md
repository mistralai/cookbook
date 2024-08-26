# Samplers
Everytime a large language model makes predictions, all of the thousands of tokens in the vocabulary are assigned some degree of probability. There are different ways you can decide to choose from those predictions. This process is known as "sampling", and there are various strategies you can use which I will cover here.

## Different Samplers

<table style="border-collapse: collapse; width: 100%; text-align: center;">
  <tr>
    <td style="font-weight:bold"><a href="temperature.md">Temperature</a></td>
    <td>
    todo
    </td>
  </tr>
  <tr>
    <td style="font-weight:bold">Top P</td>
    <td>
    todo
    </td>
  </tr>
  <tr>
    <td style="font-weight:bold">Top K</td>
    <td>
    todo
    </td>
  </tr>
  <tr>
    <td style="font-weight:bold">Min P</td>
    <td>
    todo
    </td>
  </tr>
  <tr>
    <td style="font-weight:bold">Repetition Penalty</td>
    <td>
    todo
    </td>
  </tr>
</table>

> ⚠️ You can stack Samplers one after the other, for example, you can use Top K to gather the best tokens and apply the Temperature afterwars, as well as you can do in the opposite way, the order of each sampler will impact how the final distribution will look like.
