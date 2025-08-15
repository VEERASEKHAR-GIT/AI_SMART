# BigQuery Chatbot

A simple chatbot that lets you ask questions about your data in plain English. It automatically writes SQL queries, runs them on BigQuery, and shows you the results with charts and summaries.

## What It Does

- Ask questions in normal English and get SQL queries
- Chat interface that's easy to use
- Creates charts automatically from your data
- Explains what your data means in simple words
- Browse your database tables and see what's in them
- Test many questions at once from a file
- Remembers what you asked before so you can ask follow-up questions

## What We Built With

- Streamlit for the web interface
- Google BigQuery for the database
- Azure OpenAI for the smart AI
- Pandas and NumPy for handling data
- Plotly for making charts
- Google Cloud Platform
- LangChain for connecting everything

## Problems We Solved

### 1. Making AI Understand Questions
- Problem: Hard to turn normal questions into correct SQL code
- How we fixed it: Taught the AI about our database structure
- Result: Gets the right SQL most of the time

### 2. Handling Dates
- Problem: Date formats were messy and broke our queries
- How we fixed it: Added special date handling code that doesn't break
- Result: Date questions work reliably now

### 3. Making It Work Reliably
- Problem: Sometimes queries would fail and confuse users
- How we fixed it: Added retry logic and better error messages
- Result: System keeps working even when things go wrong

### 4. Remembering Conversations
- Problem: Couldn't ask follow-up questions
- How we fixed it: Added memory so it remembers what you talked about
- Result: You can say "show me more" or "what about last year"

### 5. Smart Charts
- Problem: Hard to know what kind of chart to make
- How we fixed it: AI looks at your data and picks the best chart type
- Result: Always get charts that make sense for your data

### 6. Speed with Big Data
- Problem: Large datasets were slow and sometimes crashed
- How we fixed it: Only look at samples when needed, handle memory better
- Result: Fast even with millions of rows

### 7. Knowing When to Run SQL
- Problem: System couldn't tell if you wanted SQL or just to chat
- How we fixed it: Added smart detection for different types of messages
- Result: Responds correctly whether you want data or conversation

### 8. Managing Passwords and Keys
- Problem: Lots of passwords and API keys to manage safely
- How we fixed it: Built secure configuration system
- Result: Easy to set up and keeps credentials safe

## What You Need

- Python 3.8 or newer
- Google Cloud account with BigQuery
- Azure OpenAI account
- The packages listed in requirements.txt

## How to Set It Up

1. Get the code:
   ```bash
   git clone <repository-url>
   cd bigquery-chatbot
   ```

2. Install what you need:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Google Cloud:
   - Make a service account in Google Cloud
   - Download the key file
   - Put the file path in your settings

4. Set up Azure OpenAI:
   - Create an Azure OpenAI service
   - Set up AI models for chat and embeddings
   - Write down your endpoint, API key, and model names

## Settings File

Make a file called `.streamlit/secrets.toml` with your information:

```toml
GCP_PROJECT_ID = "your-project-name"
BIGQUERY_DATASET = "your-dataset-name"
BIGQUERY_TABLE = "your-table-name"
GOOGLE_APPLICATION_CREDENTIALS = "/path/to/your/key-file.json"

AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_ENDPOINT = "https://your-service.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2023-12-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = "your-chat-model"
EMBEDDING_MODEL = "your-embedding-model"

TEMPERATURE = "0.0"
TOP_P = "0.95"
MAX_TOKENS = "4000"
```

## How to Use It

1. Start the app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Start asking questions about your data:
   - Type questions like you would ask a person
   - Look at the SQL it creates
   - See your results and charts
   - Use the sidebar to explore your tables

## Example Questions

You can ask things like:

- "How much did we sell each year?"
- "Show me cars sold in California in 2015"
- "Which car sold the most?"
- "Compare sales in different states"
- "Show me sales over time"

## What Data Looks Like

Right now it works with car sales data that has:

- year: When the car was made
- make: Car brand like Toyota or Ford
- model: Car model like Camry or F-150
- trim: Version of the car
- body: Type like sedan or SUV
- transmission: Manual or automatic
- vin: Car ID number
- state: Where it was sold
- condition: How good the car was
- odometer: How many miles
- color: Outside color
- interior: Inside color
- seller: Who sold it
- mmr: Market value
- sellingprice: What it actually sold for
- saledate: When it was sold

## Main Features

### Charts
- Makes bar charts, line charts, pie charts, and more
- Picks the right chart type automatically
- Interactive charts you can click and zoom
- Uses your latest query results

### Data Summaries
- AI explains what your data means
- Points out important patterns
- Uses normal language, not technical terms

### Table Browser
- See all your database tables
- Look at what columns each table has
- Preview sample data
- Get statistics about your tables

### Bulk Testing
- Upload a file with many questions
- Test them all at once
- Good for checking if everything works

## When Things Go Wrong

The app handles these problems:

- Bad SQL queries
- Connection problems with BigQuery
- AI service issues
- Data format problems
- Chart creation failures

## Things to Know

- Works best with normal database tables
- Needs both Google Cloud and Azure set up correctly
- Speed depends on how complex your question is
- Some data types might not work with charts

## Getting Help

If something doesn't work:
1. Check the error message
2. Make sure your passwords and keys are right
3. Try asking your question differently
4. Make sure all the required software is installed

## License

You can use this code under the MIT License.

## Need Help?

If you have problems:
1. Read the error messages carefully
2. Check that everything is set up correctly
3. Try simpler questions first
4. Make sure your internet connection works