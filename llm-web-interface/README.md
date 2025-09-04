# LLM Web Interface

This project provides a web interface for interacting with a local Large Language Model (LLM) using FastAPI and Gradio. The application allows users to send prompts to the LLM and receive generated responses through a user-friendly interface.

## Project Structure

```
llm-web-interface
├── src
│   ├── app.py                # Entry point for the FastAPI application
│   ├── api                   # Contains API-related files
│   │   ├── __init__.py       # Package initializer
│   │   ├── llm_server.py      # Implementation of the LLM API server
│   │   └── models.py         # Data models for API requests and responses
│   ├── ui                    # Contains UI-related files
│   │   ├── __init__.py       # Package initializer
│   │   ├── gradio_app.py      # Gradio interface setup
│   │   └── components.py      # Reusable UI components
│   └── utils                 # Utility functions
│       ├── __init__.py       # Package initializer
│       └── helpers.py        # Helper functions for various tasks
├── config                    # Configuration files
│   └── config.yaml           # Application configuration settings
├── requirements.txt          # Project dependencies
├── .env.example              # Example environment variables
├── .gitignore                # Files and directories to ignore in Git
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd llm-web-interface
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Configure the application:**
   - Update the `config/config.yaml` file with the necessary settings, such as model paths.

5. **Run the application:**
   ```
   python src/app.py
   ```

6. **Access the Gradio interface:**
   - Open your web browser and navigate to `http://localhost:7860` to interact with the LLM.

## Usage Examples

- Send a prompt to the LLM and receive a generated response through the Gradio interface.
- Explore different functionalities provided by the API endpoints.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.