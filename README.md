# Wizard Chatbot - Conversational Database Agent

A sophisticated AI-powered chatbot that provides natural language interactions with databases, featuring automated data analysis, visualization, and report generation capabilities.

## 🌟 Features

### Core Functionality

- **Natural Language Database Queries**: Ask questions in plain English and get accurate data-driven answers
- **Intelligent Query Generation**: Automatically generates optimized SQL queries based on user intent
- **Real-time Data Analysis**: Performs statistical analysis and data exploration
- **Multi-format Export**: Export results to CSV, PDF reports, and visual charts

### Advanced Capabilities

- **Smart Visualization**: Automatically creates charts (pie, bar, line, histogram) based on data patterns
- **Comprehensive PDF Reports**: Generate professional reports with charts, tables, and rich text
- **Conversation Memory**: Maintains context across conversations using thread-based persistence

## 🏗️ Architecture

### Components

1. **FastAPI Backend** (`main.py`)

   - RESTful API endpoints
   - Request/response handling
   - File serving for downloads
   - Background cleanup tasks

2. **Database Agent** (`db_agent.py`)

   - Conversational AI logic
   - LangGraph-based workflow management
   - Tool orchestration
   - Message state management

3. **MCP Tools** (`mcp_tools.py`)

   - Model Context Protocol (MCP) server
   - Data visualization tools
   - Export utilities
   - Analysis functions

4. **Prompt Engineering** (`prompt_templates.py`)
   - Optimized system prompts
   - Query examples and patterns
   - Database-specific guidelines

### Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **AI/ML**: LangChain, LangGraph, Google Gemini 1.5 Pro
- **Database**: ClickHouse (configurable for other databases)
- **Visualization**: Matplotlib, ReportLab
- **Communication**: Model Context Protocol (MCP)

## 📋 Prerequisites

- Python 3.8 or higher
- ClickHouse database (or compatible SQL database)
- Google AI API key

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/somnath-wyz/brcp_chatbot.git
   cd wizard-chatbot
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Set up environment variables**

   ```bash
   cp .env.example .env
   ```

   Configure the variables in `.env`:

## 🎯 Usage

### Starting the MCP Server

```bash
uv run mcp_tools.py
```

### Starting the Fastapi Server

```bash
uv run fastapi dev
```
