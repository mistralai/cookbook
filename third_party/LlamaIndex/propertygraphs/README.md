# PropertyGraphs with LlamaIndex and MistralAI

Here, we provide cookbooks for building PropertyGraphs using LlamaIndex and MistralAI.

1.	`property_graph.ipynb` - Build a Property Graph using default extractors and retrievers.
2.	`property_graph_extractors_retrievers.ipynb` - This notebook showcases how to define different extractors, retrievers, and prompts for building PropertyGraphs. (Note: This notebook is for walkthrough purposes only and does not need to be run.)
3.	`property_graph_neo4j.ipynb` - Build PropertyGraphs with Neo4j by customizing extractors and retrievers.
4.	`property_graph_predefined_schema.ipynb` - Build PropertyGraphs with the `SchemaLLMPathExtractor` by pre-defining the schema of the PropertyGraph.
5.	`property_graph_custom_retriever.ipynb` - Build a PropertyGraph with a custom retriever using `VectorContextRetriever` and `Text2CypherRetriever`.

For more information about PropertyGraphs, refer to our [documentation](https://docs.llamaindex.ai/en/latest/examples/property_graph/graph_store/) and the [release blog post](https://www.llamaindex.ai/blog/introducing-the-property-graph-index-a-powerful-new-way-to-build-knowledge-graphs-with-llms).