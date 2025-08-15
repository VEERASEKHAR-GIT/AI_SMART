# Smart Assistant with Advanced Capabilities

A Streamlit-based intelligent document assistant that can process and answer questions about your uploaded documents using advanced AI capabilities. The application supports multiple document formats including PDF, DOCX, TXT, and CSV files.

## Features

- **Multi-format Document Support**: Upload and process PDF, DOCX, TXT, and CSV files
- **Advanced AI Models**: Choose between Azure OpenAI and Google Gemini models
- **Conversational Interface**: Interactive chat interface with conversation history
- **Document Preview**: Real-time preview of uploaded documents with special CSV table formatting
- **Source Attribution**: View sources for each AI response with page numbers
- **Vector Search**: Uses FAISS for efficient document retrieval
- **Customizable Settings**: Configurable LLM providers, models, and embedding models

## Prerequisites

- Python 3.8 or higher
- API keys for either Azure OpenAI or Google Gemini
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd smart-assistant
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in Streamlit secrets:

Create a `.streamlit/secrets.toml` file with the following structure:

```toml
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "your-azure-endpoint"
AZURE_OPENAI_API_KEY = "your-azure-api-key"
AZURE_OPENAI_DEPLOYMENT_NAME = "your-deployment-name"
AZURE_EMBEDDING_MODEL = "your-embedding-model"

# Google Gemini Configuration
GOOGLE_API_KEY = "your-google-api-key"
GEMINI_LLM_1 = "gemini-pro"
GEMINI_LLM_2 = "gemini-pro-vision"
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"

# General Settings
TEMPERATURE = "0.7"
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Configure your settings in the sidebar:
   - Select LLM Provider (Azure or Google)
   - Choose your preferred models

4. Upload your documents:
   - Supported formats: PDF, DOCX, TXT, CSV
   - Multiple files can be uploaded simultaneously
   - Use the document preview feature to inspect your files

5. Start chatting:
   - Ask questions about your uploaded documents
   - View sources for each response
   - Access your conversation history

## Project Structure

```
smart-assistant/
├── main.py                 # Main Streamlit application
├── modules/
│   ├── qa_chain.py         # Conversational QA chain setup
│   ├── retriever.py        # FAISS retriever configuration
│   ├── vector_store.py     # Vector store management
│   ├── chunker.py          # Document chunking utilities
│   ├── loader.py           # Document loading utilities
│   ├── embeddings.py       # Embedding model configuration
│   └── llm_provider.py     # LLM provider setup
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml        # Configuration secrets
└── vector_store/           # FAISS index storage (auto-created)
```

## Key Components

### Document Processing
- **Loader**: Handles multiple document formats with proper text extraction
- **Chunker**: Splits documents into manageable chunks for processing
- **Embeddings**: Converts text chunks into vector representations

### AI Integration
- **LLM Providers**: Supports both Azure OpenAI and Google Gemini
- **QA Chain**: Creates conversational retrieval chains for document Q&A
- **Vector Store**: FAISS-based vector storage for efficient similarity search

### User Interface
- **Chat Interface**: Clean, intuitive chat experience
- **Document Preview**: Enhanced preview with CSV table rendering
- **Source Attribution**: Transparent source tracking with page numbers
- **Settings Panel**: Easy model and provider configuration

## Configuration Options

### LLM Providers
- **Azure OpenAI**: Enterprise-grade OpenAI models
- **Google Gemini**: Google's advanced AI models

### Supported Models
Configure your preferred models in the secrets.toml file based on your provider subscriptions.

### Chunking Parameters
- Default chunk size: 1000 characters
- Default overlap: 200 characters
- Customizable in the code

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your API keys are correctly set in secrets.toml
2. **Model Access**: Verify you have access to the selected models
3. **Memory Issues**: For large documents, consider reducing chunk size
4. **File Upload Errors**: Check file format and size limitations

### Performance Tips

- Use smaller chunk sizes for better precision
- Increase chunk overlap for better context retention
- Consider using faster embedding models for large document sets

## Security Notes

- Keep your API keys secure in the secrets.toml file
- Don't commit secrets.toml to version control
- Use environment variables in production deployments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Your chosen license]

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Create an issue in the repository

---

**Note**: This application requires active API subscriptions to Azure OpenAI or Google Gemini services. Ensure you have proper quotas and billing set up before deployment.