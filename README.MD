# Search Validation Rules Engine

*An intelligent validation rule management system powered by Retrieval-Augmented Generation*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-2.17+-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## Overview

Search Validation Rules Engine is a sophisticated web application that combines the power of search and generation to create an intuitive validation rule management experience. Built with modern web technologies, it provides a seamless interface for discovering, analyzing, and creating validation rules through natural language interaction.

### Key Features

- **🔍 Intelligent Search** - Advanced validation rule discovery with multi-faceted filtering
- **🤖 AI-Powered Generation** - Natural language validation rule creation and modification  
- **⚡ Drag & Drop Interface** - Intuitive rule interaction and workflow
- **🎨 Modern Design** - Elegant UI with glassmorphism effects
- **📱 Responsive Layout** - Optimized for desktop and mobile experiences

---

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Search UI     │    │  Generator UI   │
│                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Filters   │  │◄──►│  │ Chat Bot  │  │
│  │ Results   │  │    │  │ Drop Zone │  │
│  └───────────┘  │    │  └───────────┘  │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
            ┌─────────────────┐
            │   RAG Engine    │
            │                 │
            │  ┌───────────┐  │
            │  │Retriever  │  │
            │  │Embeddings │  │
            │  │Generator  │  │
            │  └───────────┘  │
            └─────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd search-validation-rules-engine

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Application

```bash
python app.py
```

Navigate to `http://localhost:8050` to access the application.

---

## Project Structure

```
search-validation-rules-engine/
├── 📁 components/          # UI Components
│   ├── search.py           # Search interface & logic
│   ├── generator.py        # Chat & generation interface  
│   └── layout.py           # Main application layout
│
├── 📁 rag/                 # RAG Implementation
│   ├── retriever.py        # Search & retrieval functions
│   ├── embeddings.py       # Text embedding utilities
│   └── generator.py        # Response generation logic
│
├── 📁 utils/               # Utility Functions
│   ├── preprocessing.py    # Data preprocessing
│   └── helpers.py          # General utilities
│
├── 📁 assets/              # Static Assets
│   └── style.css           # Custom styling
│
├── 📁 data/                # Data Storage
│   ├── documents/          # Source documents
│   └── processed/          # Processed data
│
├── 📁 models/              # Model Storage
├── 📁 notebooks/           # Development Notebooks
├── app.py                  # Application entry point
└── config.py               # Configuration settings
```

---

## Usage Guide

### Search & Discovery

1. **Query Validation Rules** - Use the search box to find relevant validation rules
2. **Apply Filters** - Narrow results by category and priority
3. **Browse Results** - Review validation rule cards with descriptions and metadata

### Rule Interaction

1. **Drag to Analyze** - Drag validation rule cards to the generator drop zone
2. **Chat Interface** - Ask questions about selected validation rules
3. **Generate New Rules** - Request validation rule creation through natural language
4. **Modify Existing** - Edit validation rules using conversational commands

### Example Interactions

```
"Create a validation rule for email address format"
"Analyze the password strength validation rule"
"What's the difference between these two validation rules?"
"Modify the age validation to allow 16-65 years"
"Create a rule to validate phone number format"
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Dash + Bootstrap | Interactive web interface |
| **Styling** | Custom CSS3 | Modern glassmorphism design |
| **Icons** | Dash Iconify | Consistent iconography |
| **Backend** | Python | Core application logic |
| **AI/ML** | RAG Pipeline | Intelligent search & generation |

---

## Configuration

### Environment Variables

```bash
# Optional: Set custom configuration
export DATABASE_URL="your_database_url"
export EMBEDDING_MODEL="your_model_name"
export DEBUG="True"
```

### Custom Settings

Modify `config.py` to adjust:

- Database connections
- Embedding models
- Search parameters
- UI preferences

---

## Development

### Adding New Components

1. Create component file in `components/`
2. Import in `app.py`
3. Register callbacks
4. Add styling in `assets/style.css`

### Extending RAG Functions

1. Implement functions in `rag/` modules
2. Follow existing function signatures
3. Update imports in components
4. Test with mock data first

### Custom Styling

The application uses a modern design system with:

- **Inter font family** for typography
- **Gradient backgrounds** for visual depth
- **Glassmorphism cards** for content areas
- **Smooth animations** for interactions

---

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** thoroughly
5. **Submit** a pull request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

For questions, issues, or suggestions:

- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](link-to-issues)
- 📖 Documentation: [Project Wiki](link-to-wiki)

---

<div align="center">

**Built with ❤️ using Python & Dash**

*Making validation rule management intelligent and intuitive*

</div>


| **Column**                | **dtype**       | **Why you want it / how it’s used**                                                          | **Keep?** |
| ------------------------- | --------------- | -------------------------------------------------------------------------------------------- | --------- |
| `rule_id` (UID/hash)      | `string`        | Stable primary key—don’t rely on *name* alone.                                               | **✓**     |
| `rule_name`               | `string`        | Short handle shown to users.                                                                 | **✓**     |
| `bansta_code`             | `string`        | Domain-specific lookup key.                                                                  | **✓**     |
| `iso_code`                | `string`        | Same as above.                                                                               | **✓**     |
| `rule_body`               | `string` (long) | The full formal rule text; fed to the LLM for generation & evaluation.                       | **✓**     |
| `description_en`          | `string`        | Author-written English summary (v0).                                                         | **✓**     |
| `description_de`          | `string`        | German counterpart.                                                                          | **✓**     |
| `llm_description_en`      | `string`        | Latest LLM-generated English summary (v-LLM).                                                | **✓**     |
| `llm_description_de`      | `string`        | German LLM summary (optional but keeps things symmetric).                                    | optional  |
| `embedding`               | `object`        | *List\[float]*; same length for all rows—store raw vector or a key into your external index. | **✓**     |
| `version_major` / `minor` | `int`           | Increment when you change *body* or *logic*; helps trace breakage.                           | optional  |
| `business_division`       | `category`      | Tag for access control, dashboards, filtering (you said you’ll add this).                    | **✓**     |
| `function`                | `category`      | Another tag for retrieval clustering & facet filtering.                                      | **✓**     |
| `tags`                    | `object`        | *List\[str]*; free-form thematic labels (“payments”, “AML”, …).                              | **✓**     |
| `created_at`              | `datetime64`    | Provenance / analytics.                                                                      | optional  |
| `updated_at`              | `datetime64`    | Ditto.                                                                                       | optional  |
