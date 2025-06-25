# LightRAG - Hybrid Medical Information System

A sophisticated hybrid RAG (Retrieval-Augmented Generation) system that intelligently routes medical queries between a vector database and web search to provide accurate medical guidance for infants, children, and pregnant mothers.

#
  - Tavily Search
  - Hugging Face
  - Google API (optional)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/lightrag-medical.git
   cd lightrag-medical
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Configure databases**
   - Set up Neo4j database
   - Set up MongoDB database
   - Update connection strings in `.env`

## 🔧 Configuration

Create a `.env` file with the following variables:

```env
# API Keys
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key
HUGGINGFACE_TOKEN=your_hf_token
GOOGLE_API_KEY=your_google_key

# Database Connections
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
MONGO_URI=your_mongo_uri
MONGO_DATABASE=LightRAG

# Model Configuration
LLM_MODEL=cohere/command-a
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
EMBEDDING_MAX_TOKEN_SIZE=8192
```

## 🚀 Usage

```python
from main import app
from langchain_core.messages import HumanMessage

# Initialize the workflow
state = {
    "messages": [HumanMessage(content="my baby is vomiting for 3 days, what should I do?")]
}

# Run the workflow
for output in app.stream(state):
    for key, value in output.items():
        print(f"Node '{key}': {value}")
```

## 📁 Project Structure

```
lightrag-medical/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── README.md           # This file
└── .gitignore          # Git ignore file
```

## 🔍 How It Works

1. **Query Reception**: User submits a medical question
2. **Intelligent Routing**: LLM determines if question should go to vectorstore or web search
3. **Information Retrieval**: 
   - Vectorstore: Searches medical documents using embeddings
   - Web Search: Searches real-time information using Tavily
4. **Answer Generation**: LLM synthesizes information into coherent response
5. **Quality Check**: Answer is graded for relevance and completeness
6. **Response Delivery**: Final answer is returned to user

## 🧪 Testing

```bash
# Run the example query
python main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is designed for educational and informational purposes only. It should not replace professional medical advice. Always consult with healthcare professionals for medical decisions.

## 🆘 Support

If you encounter any issues or have questions, please open an issue on GitHub.

## 🔮 Future Enhancements

- [ ] Add more medical domains
- [ ] Implement conversation memory
- [ ] Add source citation tracking
- [ ] Improve answer quality metrics
- [ ] Add multi-language support
- [ ] Implement user feedback system 