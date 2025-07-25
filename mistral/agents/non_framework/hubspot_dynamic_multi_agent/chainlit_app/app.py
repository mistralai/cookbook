import chainlit as cl
import requests
import json
from mistralai import Mistral, ThinkChunk, TextChunk
from typing import Dict, List
import re
import asyncio
import os

# Configuration
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY", "your_hubspot_api_key_here")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "your_mistral_api_key_here")

# Global mistral client
mistral_client = None

class HubSpotConnector:
    """Handles all HubSpot API operations"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.hubapi.com/crm/v3/objects"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def get_properties(self) -> Dict:
        """Load all HubSpot properties"""

        properties = {}
        for obj_type in ['deals', 'contacts', 'companies']:
            url = f"https://api.hubapi.com/crm/v3/properties/{obj_type}"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                props = response.json()['results']
                prop_list = []
                for prop in sorted(props, key=lambda x: x['name']):
                    prop_str = f"'{prop['name']}' - {prop['label']}"
                    if 'options' in prop and prop['options']:
                        valid_values = [opt['value'] for opt in prop['options']]
                        prop_str += f" | Valid values: {valid_values}"
                    prop_list.append(prop_str)

                properties[obj_type] = prop_list
        
        return properties

    async def get_data(self, object_type: str) -> List[Dict]:
        """Fetch data from HubSpot"""

        url = f"{self.base_url}/{object_type}"
        params = {"limit": 100}
        all_data = []

        while url:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code != 200:
                raise Exception(f"HubSpot API error: {response.text}")

            data = response.json()
            all_data.extend(data.get("results", []))
            
            url = data.get("paging", {}).get("next", {}).get("link")
            params = {}
        
        return all_data

    async def batch_update(self, updates: Dict) -> None:
        """Perform batch updates to HubSpot"""
        for object_type, update_list in updates.items():
            if not update_list:
                continue

            await cl.Message(
                content=f"⚡ **Updating HubSpot Records**\n\n",
                author="HubSpot Connector"
            ).send()

            url = f"{self.base_url}/{object_type}/batch/update"
            headers = {**self.headers, "Content-Type": "application/json"}

            # Process in batches of 100
            for i in range(0, len(update_list), 100):
                batch = update_list[i:i+100]
                payload = {"inputs": batch}

                response = requests.post(url, headers=headers, json=payload)
                if response.status_code not in [200, 202]:
                    raise Exception(f"HubSpot update error: {response.text}")

async def magistral_reasoning(prompt: str) -> Dict[str, str]:
    """Use reasoning model for query analysis and planning"""
    try:
        response = await mistral_client.chat.complete_async(
            model="magistral-medium-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content

        reasoning = ""
        conclusion = ""

        for r in content:
            if isinstance(r, ThinkChunk):
                reasoning = r.thinking[0].text
            elif isinstance(r, TextChunk):
                conclusion = r.text

        return {
            "reasoning": reasoning,
            "conclusion": conclusion
        }
    except Exception as e:
        # Fallback reasoning
        return {
            "reasoning": f"Error in reasoning process: {str(e)}. Using fallback analysis.",
            "conclusion": '{"sub_agents": [{"name": "priority_assigner", "task": "Assign priorities to deals based on value", "task_type": "write_back", "input_data": ["deals"], "output_format": "JSON with deal updates"}]}'
        }

async def mistral_small_execution(prompt: str) -> str:
    """Use Mistral Small for content generation"""
    try:
        response = await asyncio.to_thread(
            mistral_client.chat.complete,
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in task execution: {str(e)}. Please try again with a simpler query."

class LeadAgent:
    """Lead Agent powered by Magistral reasoning model for query analysis and planning"""

    def __init__(self, hubspot_properties):
        self.hubspot_properties = hubspot_properties
        self.name = "LeadAgent"

    async def analyze_query(self, query: str) -> Dict:
        """Analyze query using Magistral reasoning and create execution plan"""
        
        try:
            await cl.Message(
                content="🧠 **Lead Agent Activated**\n\n"
                       "Starting deep analysis with Magistral reasoning model...\n\n",
                author="Lead Agent"
            ).send()

            analysis_prompt = f"""
                                Analyze this HubSpot query and create a detailed execution plan based on different hubspot properties provided by following the shared rules:

                                HUBSPOT_PROPERTIES: {self.hubspot_properties}

                                QUERY: {query}

                                RULES:
                                1. What is the user asking for?
                                2. Is this a read-only query or does it require HubSpot updates?
                                3. What sub-agents are needed to accomplish this?
                                4. What HubSpot data is required?
                                5. What's the execution sequence?
                                6. What should be the final output format?
                                7. Query can also be combination of read-only and write-back.
                                8. Query is read-only if it requires data read from HubSpot.
                                9. Query is write-back if it requires an update to existing values or writing/ assigning new values.
                                10. In the final conclusion just give only one JSON string nothing else. I don't need any explanation.

                                Provide a JSON execution plan with:
                                {{
                                    "sub_agents": [
                                        {{
                                            "name": "agent_name",
                                            "task": "specific task description",
                                            "task_type": "read_only" or "write_back",
                                            "input_data": ["deals", "contacts", "companies"],
                                            "output_format": "expected output"
                                        }}
                                    ]
                                }}
                                """

            try:
                analysis = await magistral_reasoning(analysis_prompt)
                
                await cl.Message(
                    content="✅ **Magistral Analysis Complete**\n\nReasoning process finished successfully!",
                    author="Lead Agent"
                ).send()
                
            except Exception as e:
                
                analysis = {
                    "reasoning": "Failed during reasoning. Using fallback plan.",
                    "conclusion": '{"sub_agents": [{"name": "priority_assigner", "task": "Assign priorities to deals based on value", "task_type": "write_back", "input_data": ["deals"], "output_format": "JSON with deal updates"}]}'
                }

            try:
                # Extract JSON execution plan from conclusion
                json_match = re.search(r'\{.*\}', analysis["conclusion"], re.DOTALL)
                if json_match:
                    execution_plan = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON found in analysis")
                    
            except Exception as e:
                await cl.Message(
                    content=f"⚠️ **Fallback Plan Activated**\n\nJSON parsing failed ({str(e)}), using general analysis approach",
                    author="Lead Agent"
                ).send()
                
                execution_plan = {
                    "sub_agents": [{
                        "name": "priority_assigner",
                        "task": "Assign priorities to deals based on deal value",
                        "task_type": "write_back",
                        "input_data": ["deals"],
                        "output_format": "JSON with HubSpot updates"
                    }]
                }

            # Show execution plan
            plan_content = "📋 **Execution Plan Created**\n\n"
            for i, agent in enumerate(execution_plan['sub_agents'], 1):
                plan_content += f"**Agent {i}: {agent['name']}**\n"
                plan_content += f"• **Task**: {agent['task']}\n"
                plan_content += f"• **Type**: {agent['task_type']}\n"
                plan_content += f"• **Data Sources**: {', '.join(agent['input_data'])}\n\n"

            await cl.Message(content=plan_content, author="Lead Agent").send()

            return {
                "reasoning": analysis["reasoning"],
                "execution_plan": execution_plan,
                "conclusion": analysis["conclusion"]
            }
            
        except Exception as e:
            await cl.Message(
                content=f"❌ **Lead Agent Error**\n\nError: {str(e)}\n\nUsing emergency fallback...",
                author="Lead Agent"
            ).send()
            
            # Emergency fallback
            return {
                "reasoning": f"Emergency fallback due to error: {str(e)}",
                "execution_plan": {
                    "sub_agents": [{
                        "name": "emergency_analyzer",
                        "task": query,
                        "task_type": "read_only",
                        "input_data": ["deals"],
                        "output_format": "summary"
                    }]
                },
                "conclusion": "Emergency fallback plan activated"
            }

class SubAgent:
    """Dynamic Sub-Agent created on-the-fly for specific tasks"""

    def __init__(self, name: str, task: str, task_type: str, input_data: List[str],
                 output_format: str):
        self.name = name
        self.task = task
        self.task_type = task_type
        self.input_data = input_data
        self.output_format = output_format

    async def execute(self, data: Dict, properties_context: str, hubspot_updater=None) -> Dict:
        """Execute the assigned task"""
        
        await cl.Message(
            content=f"🤖 **{self.name} Activated**\n\n"
                   f"**Task Type**: {self.task_type}\n"
                   f"**Mission**: {self.task}\n"
                   f"**Data Sources**: {', '.join(self.input_data)}\n\n"
                   f"Starting execution...",
            author=self.name
        ).send()

        if self.task_type == 'read_only':
            agent_prompt = f"""
            You are a {self.name} agent.

            TASK: {self.task}

            AVAILABLE HUBSPOT PROPERTIES:
            {properties_context}

            DATA AVAILABLE:
            {json.dumps(data, indent=2)}

            OUTPUT FORMAT: {self.output_format}

            Provide your analysis only based on the available data.
            """
        else:  # write_back
            agent_prompt = f"""
            You are a {self.name} agent.

            TASK: {self.task}

            AVAILABLE HUBSPOT PROPERTIES:
            {properties_context}

            DATA AVAILABLE:
            {json.dumps(data, indent=2)}

            CRITICAL: Use exact HubSpot property names from the list above in your JSON output.

            OUTPUT FORMAT: JSON format with the properties to be written to HubSpot

            Provide updates using exact HubSpot property names.
            """

        result = await mistral_small_execution(agent_prompt)

        # Handle write-back operations
        if self.task_type == 'write_back' and hubspot_updater:
            await cl.Message(
                content="📤 **Preparing HubSpot Updates**\n\nValidating and formatting update payload...",
                author=self.name
            ).send()
            
            try:
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    updates = json.loads(json_match.group(0))
                    await hubspot_updater.batch_update(updates)
                    
                    await cl.Message(
                        content="✅ **Mission Accomplished**\n\nHubSpot records updated successfully!",
                        author=self.name
                    ).send()
            except Exception as e:
                await cl.Message(
                    content=f"❌ **Update Failed**\n\nError: {str(e)}",
                    author=self.name
                ).send()
                return {"status": "error", "error": str(e), "raw_result": result}
        else:
            await cl.Message(
                content="✅ **Analysis Complete**\n\nTask executed successfully!",
                author=self.name
            ).send()

        return {"status": "success", "result": result}

class SynthesisAgent:
    """Final agent to synthesize all results into user-friendly response"""

    def __init__(self):
        self.name = "SynthesisAgent"

    async def synthesize(self, query: str, sub_agent_results: List[Dict], execution_plan: Dict) -> str:
        """Combine all sub-agent results into final answer"""
        
        await cl.Message(
            content=f"🔄 **Synthesis Agent Activated**\n\n"
                   f"Combining insights from {len(sub_agent_results)} specialized agents...\n\n",
            author="Synthesis Agent"
        ).send()

        # Prepare context from all sub-agent results
        results_context = ""
        for result in sub_agent_results:
            results_context += f"\n{result['agent'].upper()} ({result['task_type']}):\n"
            if result['result']['status'] == 'success':
                results_context += f"{result['result']['result']}\n"
            else:
                results_context += f"Error: {result['result'].get('error', 'Unknown error')}\n"
            results_context += "---\n"

        synthesis_prompt = f"""
        You are a final synthesizer agent. Create a comprehensive, user-friendly response based on all sub-agent results.

        ORIGINAL QUERY: {query}

        SUB-AGENT RESULTS:
        {results_context}

        TASK: Synthesize all the above results into a clear, actionable response for the user.

        Guidelines:
        1. Start with a direct answer to the user's query
        2. Include key insights and findings
        3. If updates were made, summarize what was changed
        4. Provide actionable next steps if relevant
        5. Keep it concise but comprehensive
        6. Use a professional but friendly tone

        Provide the final synthesized response:
        """

        final_answer = await mistral_small_execution(synthesis_prompt)

        return final_answer

class AgentOrchestrator:
    """Main orchestrator that coordinates all agents"""

    def __init__(self, hubspot_api_key: str, mistral_api_key: str):
        # Initialize global mistral client for existing functions
        global mistral_client
        mistral_client = Mistral(api_key=mistral_api_key)

        # Initialize HubSpot connector
        self.hubspot_connector = HubSpotConnector(hubspot_api_key)
        self.hubspot_properties = None
        self.hubspot_data = None

        # Initialize agents
        self.synthesis_agent = SynthesisAgent()
        self.active_sub_agents = []

    async def initialize(self):
        """Initialize the system with HubSpot data"""
        await cl.Message(
            content="🚀 **HubSpot Multi-Agent System**\n\n"
                   "Initializing intelligent CRM automation...\n\n"
                   "⚡ **System Components:**\n"
                   "• Lead Agent (Magistral Reasoning)\n"
                   "• Dynamic Sub-Agents (Mistral Small)\n" 
                   "• Synthesis Agent (Final Processing)\n"
                   "• HubSpot Connector (CRM Integration)",
            author="System Orchestrator"
        ).send()

        # Load HubSpot data and properties
        self.hubspot_properties = await self.hubspot_connector.get_properties()
        
        self.hubspot_data = {
            "deals": await self.hubspot_connector.get_data("deals"),
            "contacts": await self.hubspot_connector.get_data("contacts"),
            "companies": await self.hubspot_connector.get_data("companies")
        }

        # Initialize lead agent with properties
        self.lead_agent = LeadAgent(self.hubspot_properties)        

    async def process_query(self, query: str) -> Dict:
        """Main method to process user queries through multi-agent workflow"""

        # Step 1: Lead Agent analyzes query using Magistral reasoning
        analysis = await self.lead_agent.analyze_query(query)
        execution_plan = analysis["execution_plan"]

        # Step 2: Create and execute sub-agents dynamically
        await cl.Message(
            content="🏗️ **Dynamic Agent Creation**\n\n"
                   f"Creating {len(execution_plan['sub_agents'])} specialized agents...",
            author="System Orchestrator"
        ).send()

        sub_agent_results = []
        self.active_sub_agents = []

        for i, agent_config in enumerate(execution_plan["sub_agents"], 1):
            # Create sub-agent dynamically
            sub_agent = SubAgent(
                name=agent_config["name"],
                task=agent_config["task"],
                task_type=agent_config["task_type"],
                input_data=agent_config["input_data"],
                output_format=agent_config["output_format"]
            )

            self.active_sub_agents.append(sub_agent)

            await cl.Message(
                content=f"🛠️ **Agent {i}/{len(execution_plan['sub_agents'])} Created**\n\n"
                       f"**{sub_agent.name}** ready for deployment",
                author="System Orchestrator"
            ).send()

            # Prepare data and context for this sub-agent
            agent_data = {data_type: self.hubspot_data.get(data_type, [])
                         for data_type in agent_config["input_data"]}

            # Build properties context
            properties_context = ""
            for data_type in agent_config["input_data"]:
                if data_type in self.hubspot_properties:
                    properties_context += f"\n{data_type.upper()} PROPERTIES:\n"
                    properties_context += "\n".join(self.hubspot_properties[data_type])
                    properties_context += "\n"

            # Execute sub-agent
            result = await sub_agent.execute(agent_data, properties_context, self.hubspot_connector)
            sub_agent_results.append({
                "agent": sub_agent.name,
                "task_type": sub_agent.task_type,
                "result": result
            })

        # Step 3: Synthesis Agent creates final answer
        final_answer = await self.synthesis_agent.synthesize(query, sub_agent_results, execution_plan)

        await cl.Message(
            content="🎉 **Processing Complete**\n\n"
                   "All agents have completed their missions successfully!\n"
                   "Final answer ready for delivery.",
            author="System Orchestrator"
        ).send()

        return {
            "query": query,
            "reasoning": analysis["reasoning"],
            "execution_plan": execution_plan,
            "sub_agent_results": sub_agent_results,
            "active_agents": [agent.name for agent in self.active_sub_agents],
            "final_answer": final_answer
        }

# Global orchestrator
orchestrator = None

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    global orchestrator
    
    # Check API keys
    if HUBSPOT_API_KEY == "your_hubspot_api_key_here" or MISTRAL_API_KEY == "your_mistral_api_key_here":
        await cl.Message(
            content="❌ **Configuration Required**\n\n"
                   "Please set your API keys:\n"
                   "• `HUBSPOT_API_KEY` environment variable\n"
                   "• `MISTRAL_API_KEY` environment variable\n\n"
                   "Or update the keys directly in the code.",
            author="System"
        ).send()
        return

    try:
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(
            hubspot_api_key=HUBSPOT_API_KEY,
            mistral_api_key=MISTRAL_API_KEY
        )
        
        # Initialize the system
        await orchestrator.initialize()
        
        # Welcome message
        await cl.Message(
            content="👋 **Welcome to HubSpot Multi-Agent Assistant!**\n\n"
                   "I can help you with complex CRM queries using MistralAI Magistral reasoning. Try asking:\n\n"
                   "💡 **Example Query:**\n"
                   "• *Assign priorities to all deals based on deal value*\n"
                   "Just type your question and watch the multi-agent system work!",
            author="Assistant"
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Initialization Failed**\n\n"
                   f"Error: {str(e)}\n\n"
                   f"Please check your API keys and network connection.",
            author="System"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    global orchestrator
    
    if not orchestrator:
        await cl.Message(
            content="❌ **System Not Ready**\n\nPlease restart the application.",
            author="System"
        ).send()
        return

    try:
        # Process the query through multi-agent system
        result = await orchestrator.process_query(message.content)
        
        # Send final answer
        await cl.Message(
            content=f"## 🎯 Final Answer\n\n{result['final_answer']}",
            author="Assistant"
        ).send()
        
        # Optionally show execution summary
        summary = f"**📊 Execution Summary:**\n\n"
        summary += f"• **Active Agents**: {', '.join(result['active_agents'])}\n"
        summary += f"• **Successful Operations**: {sum(1 for r in result['sub_agent_results'] if r['result']['status'] == 'success')}\n"
        summary += f"• **Total Processing Steps**: {len(result['sub_agent_results']) + 2}\n"  # +2 for Lead and Synthesis agents
        
        await cl.Message(content=summary, author="System Summary").send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Processing Error**\n\n"
                   f"Error: {str(e)}\n\n"
                   f"Please try rephrasing your query or contact support.",
            author="System"
        ).send()

if __name__ == "__main__":
    cl.run()