from typing import Literal, Optional, Dict, Any, List, AsyncGenerator
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.runnables import RunnableConfig
from datetime import datetime
import logging
from pathlib import Path
from prompt_templates import system_prompt
from mcp import ClientSession, StdioServerParameters
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)

class DatabaseAgent:
    """
    A conversational database agent that makes database interactions transparent to users.
    Users interact as if talking to a normal AI assistant, while the system handles
    database queries and file exports intelligently in the background.
    Enhanced with column meanings for ClickHouse legacy databases.
    """
    
    def __init__(self, llm: BaseChatModel, db: SQLDatabase, checkpointer=None, export_directory: str = "exports") -> None:
        """
        Initialize the ConversationalDatabaseAgent with column meanings support.
        
        Args:
            llm: The language model for conversation and query generation
            db: SQLDatabase instance for database operations
            checkpointer: Optional checkpointer for conversation persistence
            export_directory: Directory to store exported files
            column_meanings: Dictionary mapping column names to their meanings
        """
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.checkpointer = checkpointer

        self.llm = llm
        self.db = db
        self.export_dir = Path(export_directory)
        self.export_dir.mkdir(exist_ok=True)
        
        # Database toolkit
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.db_tools = self.toolkit.get_tools()

        # Create custom tools that the LLM can use
        self.mcp_tools = []
        
        # Combine all tools
        self.all_tools = []
        
    async def connect_to_mcp_server(self, server_script_path: str = "mcp_tools.py"):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")

        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        
        command = "python" if is_python else "node"
        
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params)) # type: ignore
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        self.mcp_tools = await load_mcp_tools(self.session)
        self.all_tools = self.db_tools + self.mcp_tools
        
        self._build_graph(self.checkpointer)
    
    def _build_graph(self, checkpointer):
        """Build the conversational graph."""
        self.builder = StateGraph(MessagesState)
        
        # Add the main conversation node
        self.builder.add_node("conversation", self.handle_conversation)
        
        # Add tool execution node
        self.tool_node = ToolNode(self.all_tools)
        self.builder.add_node("tools", self.tool_node)
        
        # Define the flow
        self.builder.add_edge(START, "conversation")
        self.builder.add_conditional_edges(
            "conversation",
            self.should_use_tools,
            {
                "tools": "tools",
                "end": END
            }
        )
        self.builder.add_edge("tools", "conversation")
        
        self.agent = self.builder.compile(checkpointer=checkpointer)
    
    async def handle_conversation(self, state: MessagesState) -> Dict[str, List[BaseMessage]]:
        """
        Main conversation handler that makes the agent feel like a normal AI assistant.
        """
        try:
            # Enhanced system prompt for conversational AI with column meanings
            prompt = system_prompt.format(dialect=self.db.dialect, tables=self.db.get_table_names(), date=datetime.now().strftime('%Y-%m-%d'))
            messages = [SystemMessage(content=prompt)] + state["messages"]
            
            # Use LLM with tools
            llm_with_tools = self.llm.bind_tools(self.all_tools)
            response = await llm_with_tools.ainvoke(messages)
            
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error in conversation handling: {e}")
            error_response = AIMessage(
                content="I apologize, but I encountered an issue while processing your request. Could you please try again?"
            )
            return {"messages": [error_response]}
    
    def should_use_tools(self, state: MessagesState) -> Literal["tools", "end"]:
        """Determine if tools should be used based on the last message."""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls: # type: ignore
            return "tools"
        else:
            return "end"
    
    async def stream(
        self, 
        message: str, 
        thread_id: Optional[str] = None,
        include_metadata: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a conversational interaction with real-time updates.
        
        Args:
            message: User's message
            thread_id: Optional thread ID for conversation continuity
            include_metadata: Whether to include metadata in stream events
            
        Yields:
            Dictionary with streaming events including:
            - event_type: Type of event ('start', 'node', 'edge', 'tool', 'message', 'end', 'error')
            - data: Event-specific data
            - metadata: Optional metadata about the event
        """
        try:
            # Configure for thread persistence if provided
            config: Optional[RunnableConfig] = None
            if thread_id:
                config = {"configurable": {"thread_id": thread_id}}
            
            logger.info(f"Starting streaming for message: {message[:100]}...")
            
            # Yield start event
            yield {
                "event_type": "start",
                "data": {
                    "message": message,
                    "thread_id": thread_id,
                    "timestamp": datetime.now().isoformat()
                },
                "metadata": {"agent": "ConversationalDatabaseAgent"} if include_metadata else None
            }
            
            created_files = []
            final_response = ""
            
            # Stream through the graph execution
            async for event in self.agent.astream(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                stream_mode="values"  # Stream intermediate values
            ):
                try:
                    # Extract current messages
                    current_messages = event.get("messages", [])
                    
                    if current_messages:
                        latest_message = current_messages[-1]
                        
                        # Handle different message types
                        if isinstance(latest_message, AIMessage):
                            # Check if this is a tool call
                            if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                                for tool_call in latest_message.tool_calls:
                                    yield {
                                        "event_type": "tool",
                                        "data": {
                                            "tool_name": tool_call.get("name", "unknown"),
                                            "tool_args": tool_call.get("args", {}),
                                            "tool_id": tool_call.get("id", "")
                                        },
                                        "metadata": {
                                            "timestamp": datetime.now().isoformat()
                                        } if include_metadata else None
                                    }
                            
                            # Check if this is a regular AI response
                            elif latest_message.content and latest_message.content != final_response:
                                final_response = latest_message.content
                                
                                yield {
                                    "event_type": "message",
                                    "data": {
                                        "content": latest_message.content,
                                        "message_type": "ai",
                                        "partial": False
                                    },
                                    "metadata": {
                                        "timestamp": datetime.now().isoformat()
                                    } if include_metadata else None
                                }
                        
                        # Handle tool messages (results from tool execution)
                        elif hasattr(latest_message, 'content') and "Download:" in str(latest_message.content):
                            created_files.append(str(latest_message.content))
                            
                            yield {
                                "event_type": "file_created",
                                "data": {
                                    "file_info": str(latest_message.content),
                                    "tool_result": True
                                },
                                "metadata": {
                                    "timestamp": datetime.now().isoformat()
                                } if include_metadata else None
                            }
                
                except Exception as stream_error:
                    logger.warning(f"Error processing stream event: {stream_error}")
                    yield {
                        "event_type": "warning",
                        "data": {
                            "message": f"Stream processing warning: {str(stream_error)}"
                        },
                        "metadata": {
                            "timestamp": datetime.now().isoformat()
                        } if include_metadata else None
                    }
            
            # Yield final completion event
            yield {
                "event_type": "end",
                "data": {
                    "final_response": final_response,
                    "created_files": created_files,
                    "thread_id": thread_id,
                    "timestamp": datetime.now().isoformat()
                },
                "metadata": {
                    "total_files_created": len(created_files),
                    "processing_complete": True
                } if include_metadata else None
            }
            
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield {
                "event_type": "error",
                "data": {
                    "error": str(e),
                    "thread_id": thread_id,
                    "timestamp": datetime.now().isoformat()
                },
                "metadata": {
                    "error_type": type(e).__name__
                } if include_metadata else None
            }
    
    async def run(
            self, 
            message: str, 
            last_file_request_message_index_map: dict[str, int], 
            thread_id: Optional[str] = None
        ) -> Dict[str, Any]:
        """
        Run a conversational interaction.
        
        Args:
            message: User's message
            thread_id: Optional thread ID for conversation continuity
            
        Returns:
            Dictionary with response and any generated files
        """
        try:
            # Configure for thread persistence if provided
            config: Optional[RunnableConfig] = None
            
            if thread_id:
                config = {"configurable": {"thread_id": thread_id}}
            
            # Process the conversation
            response = await self.agent.ainvoke({"messages": [HumanMessage(content=message)]}, config=config)
            
            # Extract the final response
            final_message = response["messages"][-1]
            
            # Check if any files were created by looking at tool responses
            created_files = []

            if thread_id:
                last_file_request_message_index = last_file_request_message_index_map[thread_id] if thread_id in last_file_request_message_index_map else 0
                unseen_messages = response["messages"][(len(response["messages"]) - last_file_request_message_index) - 1]
                
                file_request_msg_idx = last_file_request_message_index
                for i, msg in enumerate(unseen_messages):
                    if hasattr(msg, 'content') and msg.content:
                        if "Download:" in msg.content:
                            # Extract download info from tool responses
                            created_files.append(msg.content)
                            file_request_msg_idx = i

                if file_request_msg_idx == last_file_request_message_index:
                    last_file_request_message_index_map[thread_id] = len(response["messages"])
                else:
                    last_file_request_message_index_map[thread_id] = file_request_msg_idx + last_file_request_message_index + 1

            result = {
                "response": final_message.content,
                "created_files": created_files,
                "thread_id": thread_id
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in conversational run: {e}")
            return {
                "response": "I apologize, but I encountered an issue. Could you please try again?",
                "created_files": [],
                "thread_id": thread_id,
                "error": str(e)
            }
    
    def cleanup_old_files(self, hours: int = 24) -> int:
        """Remove files older than specified hours."""
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        deleted_count = 0
        
        for file_path in self.export_dir.glob("*"):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Cleaned up old file: {file_path.name}")
        
        return deleted_count
    
    async def close_mcp_server(self):
        """
        Clean up resources and close MCP connection.
        """
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
            logger.info("MCP connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing MCP connection: {e}")