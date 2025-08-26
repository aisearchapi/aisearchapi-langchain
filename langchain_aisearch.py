"""
🚀 LangChain Integration for AI Search API
==========================================

A powerful integration that brings AI Search API's intelligent search capabilities 
into the LangChain ecosystem. Search, analyze, and build AI applications with 
context-aware responses and source attribution.

Author: AI Search API Team
Version: 1.0.0
License: MIT
"""

import os
import json
from typing import Any, List, Optional, Dict, Iterator, AsyncIterator
from dataclasses import dataclass, field

# LangChain imports
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage,
    ChatGeneration,
    Generation,
    LLMResult,
    ChatResult
)
from langchain.schema.messages import BaseMessageChunk, AIMessageChunk
from langchain.schema.output import ChatGenerationChunk
from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import Field, root_validator, BaseModel
from langchain.tools import BaseTool, StructuredTool
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate

# Your AI Search API client
from aisearchapi import AISearchAPIClient, ChatMessage, AISearchAPIError


# ============================================================================
# 🎯 Core LangChain LLM Implementation
# ============================================================================

class AISearchLLM(LLM):
    """
    🤖 LangChain LLM wrapper for AI Search API
    
    This class integrates AI Search API as a Large Language Model in LangChain,
    enabling you to use intelligent search capabilities in any LangChain workflow.
    
    Features:
    - Semantic search with context awareness
    - Source attribution for transparency
    - Markdown/text response formatting
    - Built-in error handling and retries
    
    Example:
        ```python
        llm = AISearchLLM(api_key="your-key")
        response = llm("What is quantum computing?")
        ```
    """
    
    api_key: str = Field(default=None, description="AI Search API key")
    base_url: str = Field(default="https://api.aisearchapi.io", description="API base URL")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    response_type: str = Field(default="markdown", description="Response format: 'text' or 'markdown'")
    include_sources: bool = Field(default=True, description="Include sources in response")
    client: Optional[AISearchAPIClient] = Field(default=None, exclude=True)
    
    class Config:
        """Pydantic config"""
        extra = "forbid"
        arbitrary_types_allowed = True
    
    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate API key from environment or direct input"""
        api_key = values.get("api_key") or os.getenv("AI_SEARCH_API_KEY")
        if not api_key:
            raise ValueError(
                "AI Search API key not found. Please set 'api_key' or "
                "environment variable 'AI_SEARCH_API_KEY'"
            )
        values["api_key"] = api_key
        return values
    
    def __init__(self, **kwargs):
        """Initialize the AI Search LLM"""
        super().__init__(**kwargs)
        self.client = AISearchAPIClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM"""
        return "ai_search_api"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return parameters that identify this LLM"""
        return {
            "base_url": self.base_url,
            "response_type": self.response_type,
            "include_sources": self.include_sources,
            "timeout": self.timeout
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call AI Search API with the given prompt
        
        Args:
            prompt: The search query
            stop: Stop sequences (not used by AI Search API)
            run_manager: Callback manager for tracking
            **kwargs: Additional parameters
            
        Returns:
            The search response with optional sources
        """
        try:
            # Extract context from kwargs if provided
            context = kwargs.get("context", [])
            
            # Convert context to ChatMessage format if needed
            chat_context = []
            if context:
                for msg in context:
                    if isinstance(msg, str):
                        chat_context.append(ChatMessage(role="user", content=msg))
                    elif isinstance(msg, dict):
                        chat_context.append(ChatMessage(role="user", content=msg.get("content", "")))
            
            # Make the API call
            response = self.client.search(
                prompt=prompt,
                context=chat_context if chat_context else None,
                response_type=self.response_type
            )
            
            # Format the response
            result = response.answer
            
            if self.include_sources and response.sources:
                result += "\n\n📚 **Sources:**\n"
                for i, source in enumerate(response.sources, 1):
                    result += f"{i}. {source}\n"
            
            # Add metadata as a comment (useful for debugging)
            result += f"\n<!-- Processing time: {response.total_time}ms -->"
            
            return result
            
        except AISearchAPIError as e:
            return f"❌ AI Search API Error: {e.description}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async version of _call (currently uses sync implementation)"""
        # For true async, you'd need an async version of the client
        return self._call(prompt, stop, run_manager, **kwargs)
    
    def get_num_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    def save(self, file_path: str) -> None:
        """Save the LLM configuration"""
        config = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "response_type": self.response_type,
            "include_sources": self.include_sources
        }
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> "AISearchLLM":
        """Load LLM configuration from file"""
        with open(file_path, 'r') as f:
            config = json.load(f)
        return cls(**config)


# ============================================================================
# 💬 Chat Model Implementation
# ============================================================================

class AISearchChat(BaseChatModel):
    """
    🗨️ Chat-optimized wrapper for AI Search API
    
    This chat model maintains conversation context and provides a more
    natural chat interface for the AI Search API.
    
    Features:
    - Automatic context management
    - Message history tracking
    - Role-based messaging support
    - Streaming capabilities (simulated)
    
    Example:
        ```python
        chat = AISearchChat(api_key="your-key")
        messages = [
            HumanMessage(content="Tell me about solar energy"),
            HumanMessage(content="What are the main advantages?")
        ]
        response = chat(messages)
        ```
    """
    
    api_key: str = Field(default=None)
    base_url: str = Field(default="https://api.aisearchapi.io")
    timeout: int = Field(default=30)
    response_type: str = Field(default="markdown")
    include_sources: bool = Field(default=True)
    streaming: bool = Field(default=False)
    client: Optional[AISearchAPIClient] = Field(default=None, exclude=True)
    
    class Config:
        """Pydantic config"""
        extra = "forbid"
        arbitrary_types_allowed = True
    
    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate API key"""
        api_key = values.get("api_key") or os.getenv("AI_SEARCH_API_KEY")
        if not api_key:
            raise ValueError("AI Search API key required")
        values["api_key"] = api_key
        return values
    
    def __init__(self, **kwargs):
        """Initialize the chat model"""
        super().__init__(**kwargs)
        self.client = AISearchAPIClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    @property
    def _llm_type(self) -> str:
        """Return identifier"""
        return "ai_search_chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response from messages
        
        Args:
            messages: List of chat messages
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional parameters
            
        Returns:
            ChatResult with the response
        """
        # Extract the last message as the prompt
        if not messages:
            raise ValueError("No messages provided")
        
        last_message = messages[-1]
        prompt = last_message.content
        
        # Convert previous messages to context
        context = []
        for msg in messages[:-1]:
            if isinstance(msg, (HumanMessage, SystemMessage)):
                context.append(ChatMessage(role="user", content=msg.content))
        
        try:
            # Make the API call
            response = self.client.search(
                prompt=prompt,
                context=context if context else None,
                response_type=self.response_type
            )
            
            # Format the response
            content = response.answer
            
            if self.include_sources and response.sources:
                content += "\n\n📚 **Sources:**\n"
                for i, source in enumerate(response.sources, 1):
                    content += f"{i}. {source}\n"
            
            # Create the response message
            message = AIMessage(
                content=content,
                additional_kwargs={
                    "sources": response.sources,
                    "processing_time_ms": response.total_time
                }
            )
            
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except AISearchAPIError as e:
            error_message = AIMessage(content=f"❌ Error: {e.description}")
            generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[generation])
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation (uses sync for now)"""
        return self._generate(messages, stop, run_manager, **kwargs)
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream responses (simulated streaming)
        
        Note: AI Search API doesn't support streaming, so this simulates it
        by yielding the response in chunks.
        """
        # Get the full response
        result = self._generate(messages, stop, run_manager, **kwargs)
        full_content = result.generations[0].message.content
        
        # Simulate streaming by breaking into chunks
        chunk_size = 20  # Characters per chunk
        for i in range(0, len(full_content), chunk_size):
            chunk_text = full_content[i:i + chunk_size]
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=chunk_text)
            )
            yield chunk
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async streaming (simulated)"""
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            yield chunk


# ============================================================================
# 🛠️ Custom Tools for AI Search
# ============================================================================

class AISearchTool(BaseTool):
    """
    🔍 AI Search as a LangChain Tool
    
    Use AI Search API as a tool in agents and chains for intelligent
    information retrieval.
    
    Example:
        ```python
        tool = AISearchTool(api_key="your-key")
        result = tool.run("Latest developments in quantum computing")
        ```
    """
    
    name: str = "ai_search"
    description: str = (
        "Intelligent search tool that provides accurate, sourced answers. "
        "Use this when you need to find information, research topics, or "
        "answer questions with reliable sources."
    )
    api_key: str = Field(default=None)
    client: Optional[AISearchAPIClient] = Field(default=None, exclude=True)
    
    class Config:
        """Pydantic config"""
        extra = "forbid"
        arbitrary_types_allowed = True
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the tool"""
        api_key = api_key or os.getenv("AI_SEARCH_API_KEY")
        if not api_key:
            raise ValueError("API key required")
        
        super().__init__(api_key=api_key, **kwargs)
        self.client = AISearchAPIClient(api_key=self.api_key)
    
    def _run(self, query: str, run_manager: Optional[Any] = None) -> str:
        """Execute a search"""
        try:
            response = self.client.search(prompt=query, response_type="markdown")
            
            result = f"🔍 **Search Results:**\n\n{response.answer}\n\n"
            if response.sources:
                result += "📚 **Sources:**\n"
                for i, source in enumerate(response.sources, 1):
                    result += f"{i}. {source}\n"
            
            return result
        except Exception as e:
            return f"❌ Search failed: {str(e)}"
    
    async def _arun(self, query: str, run_manager: Optional[Any] = None) -> str:
        """Async execution"""
        return self._run(query, run_manager)


def create_balance_tool(api_key: Optional[str] = None) -> Tool:
    """
    💰 Create a tool to check AI Search API balance
    
    Example:
        ```python
        balance_tool = create_balance_tool(api_key="your-key")
        credits = balance_tool.run("check")
        ```
    """
    api_key = api_key or os.getenv("AI_SEARCH_API_KEY")
    client = AISearchAPIClient(api_key=api_key)
    
    def check_balance(input: str = "") -> str:
        """Check account balance"""
        try:
            balance = client.balance()
            return (
                f"💳 **Account Balance:**\n"
                f"Available Credits: {balance.available_credits:,}\n"
                f"{'⚠️ Low balance warning!' if balance.available_credits < 10 else '✅ Balance healthy'}"
            )
        except Exception as e:
            return f"❌ Failed to check balance: {str(e)}"
    
    return Tool(
        name="check_ai_search_balance",
        func=check_balance,
        description="Check AI Search API account balance and available credits"
    )


# ============================================================================
# 🎭 Pre-built Chains and Agents
# ============================================================================

def create_research_chain(api_key: Optional[str] = None) -> LLMChain:
    """
    📚 Create a research chain with AI Search
    
    This chain is optimized for research tasks, providing detailed
    answers with sources.
    
    Example:
        ```python
        chain = create_research_chain(api_key="your-key")
        result = chain.run("Explain the latest CRISPR developments")
        ```
    """
    llm = AISearchLLM(api_key=api_key, response_type="markdown", include_sources=True)
    
    prompt = PromptTemplate(
        input_variables=["topic"],
        template=(
            "Please provide a comprehensive research summary on the following topic:\n\n"
            "Topic: {topic}\n\n"
            "Include:\n"
            "1. Key concepts and definitions\n"
            "2. Current state of knowledge\n"
            "3. Recent developments\n"
            "4. Future implications\n"
            "5. Reliable sources\n"
        )
    )
    
    return LLMChain(llm=llm, prompt=prompt, verbose=True)


def create_qa_chain(api_key: Optional[str] = None) -> ConversationChain:
    """
    ❓ Create a Q&A chain with memory
    
    This chain maintains conversation context for follow-up questions.
    
    Example:
        ```python
        qa = create_qa_chain(api_key="your-key")
        qa.run("What is machine learning?")
        qa.run("What are its main applications?")  # Remembers context
        ```
    """
    llm = AISearchLLM(api_key=api_key, response_type="text")
    memory = ConversationBufferMemory()
    
    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )


def create_fact_checker_chain(api_key: Optional[str] = None) -> LLMChain:
    """
    ✅ Create a fact-checking chain
    
    Specialized chain for verifying claims and statements.
    
    Example:
        ```python
        checker = create_fact_checker_chain(api_key="your-key")
        result = checker.run("The Earth is flat")
        ```
    """
    llm = AISearchLLM(api_key=api_key, response_type="markdown", include_sources=True)
    
    prompt = PromptTemplate(
        input_variables=["claim"],
        template=(
            "Please fact-check the following claim:\n\n"
            "Claim: {claim}\n\n"
            "Provide:\n"
            "1. Verdict: TRUE / FALSE / PARTIALLY TRUE / UNVERIFIABLE\n"
            "2. Explanation with evidence\n"
            "3. Reliable sources supporting the verdict\n"
        )
    )
    
    return LLMChain(llm=llm, prompt=prompt, verbose=True)


# ============================================================================
# 🎯 Utility Functions
# ============================================================================

def test_connection(api_key: Optional[str] = None) -> bool:
    """
    🔌 Test the AI Search API connection
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        api_key = api_key or os.getenv("AI_SEARCH_API_KEY")
        client = AISearchAPIClient(api_key=api_key)
        balance = client.balance()
        print(f"✅ Connection successful! Credits: {balance.available_credits:,}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False


def estimate_cost(text: str, cost_per_credit: float = 0.001) -> float:
    """
    💵 Estimate the cost of a search query
    
    Args:
        text: The search query
        cost_per_credit: Cost per API credit (default: $0.001)
        
    Returns:
        Estimated cost in dollars
    """
    # Rough estimate: 1 credit per 100 characters
    credits = max(1, len(text) // 100)
    return credits * cost_per_credit


# ============================================================================
# 🚀 Quick Start Examples
# ============================================================================

def example_basic_usage():
    """Basic usage example"""
    print("🎯 Basic AI Search LangChain Usage\n" + "="*40)
    
    # Initialize the LLM
    llm = AISearchLLM(api_key="your-api-key-here")
    
    # Simple query
    response = llm("What are the benefits of renewable energy?")
    print(f"Response:\n{response}\n")
    
    # With context
    response = llm(
        "What are the main challenges?",
        context=["We're discussing solar panel installation for homes"]
    )
    print(f"Contextual Response:\n{response}")


def example_chat_usage():
    """Chat model usage example"""
    print("💬 Chat Model Usage\n" + "="*40)
    
    # Initialize chat model
    chat = AISearchChat(api_key="your-api-key-here")
    
    # Create a conversation
    messages = [
        HumanMessage(content="Tell me about electric vehicles"),
        HumanMessage(content="What about their environmental impact?"),
        HumanMessage(content="How do they compare to hydrogen cars?")
    ]
    
    # Get response
    response = chat(messages)
    print(f"Chat Response:\n{response.content}")


def example_agent_usage():
    """Agent with tools example"""
    print("🤖 Agent with AI Search Tools\n" + "="*40)
    
    from langchain.agents import initialize_agent, AgentType
    
    # Create tools
    search_tool = AISearchTool(api_key="your-api-key-here")
    balance_tool = create_balance_tool(api_key="your-api-key-here")
    
    # Create agent
    llm = AISearchLLM(api_key="your-api-key-here")
    agent = initialize_agent(
        tools=[search_tool, balance_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Run agent
    result = agent.run("Search for information about Mars colonization and check my balance")
    print(f"Agent Result:\n{result}")


def example_chain_usage():
    """Custom chains example"""
    print("🔗 Custom Chain Usage\n" + "="*40)
    
    # Research chain
    research_chain = create_research_chain(api_key="your-api-key-here")
    result = research_chain.run("Quantum computing applications in medicine")
    print(f"Research Result:\n{result}\n")
    
    # Fact checker
    fact_checker = create_fact_checker_chain(api_key="your-api-key-here")
    result = fact_checker.run("Coffee is the world's second-most traded commodity")
    print(f"Fact Check Result:\n{result}")


if __name__ == "__main__":
    print("""
    🚀 AI Search API + LangChain Integration
    ========================================
    
    This module provides complete LangChain integration for AI Search API.
    
    Quick Start:
    1. Set your API key: export AI_SEARCH_API_KEY='your-key'
    2. Import the components you need
    3. Start building amazing AI applications!
    
    Run the examples to see it in action!
    """)
    
    # Uncomment to run examples:
    # example_basic_usage()
    # example_chat_usage()
    # example_agent_usage()
    # example_chain_usage()
