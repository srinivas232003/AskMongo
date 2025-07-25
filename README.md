# Streamlit MongoDB LLaMA 3 App

This project is a Streamlit application that allows users to interact with a MongoDB database using natural language queries. The application leverages LLaMA 3 via Groq to convert user queries into MongoDB queries and provides insightful summaries based on the results.

## Project Structure

```
streamlit-mongo-llama3-app
├── app.py               # Main entry point for the Streamlit application
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd streamlit-mongo-llama3-app
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**
   Create a `.env` file in the root directory and add your MongoDB connection string and any other necessary credentials.

5. **Run the application:**
   ```bash
   streamlit run src/app.py
   ```

## Usage

- Enter a natural language query in the chat input.
- Use the optional filters to refine your query.
- The application will display results as a table if the output is large or tabular.
- If the output is short or insightful, it will provide a summary generated by LLaMA 3.

## Example Queries

- "Show me all crops grown in 2023"
- "What kind of crops are most risky in summer?"
- "What does the risk_level field represent?"
- "Give a summary of yield trends in 2022 and 2023"

