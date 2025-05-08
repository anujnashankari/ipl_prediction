# IPL Cricket Prediction System

A comprehensive machine learning system for predicting IPL cricket match outcomes, scores, and player performance.

## Features

- Machine learning models for predicting:
  - Match winners
  - Team scores
  - Player performance
- LLM-based reasoning for prediction explanations
- Django and FastAPI backends with RESTful APIs
- Data collection pipeline using web scraping
- Comprehensive database design for cricket statistics

## Setup Instructions

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

### Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/ipl-prediction-system.git
cd ipl-prediction-system
\`\`\`

2. Create a virtual environment and install dependencies:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

3. Set up the database:
\`\`\`bash
cd django_backend
python manage.py migrate
python manage.py createsuperuser
\`\`\`

4. Start the Ollama server for LLM integration:
\`\`\`bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull llama2
\`\`\`

### Running the Application

#### Using Docker

\`\`\`bash
docker-compose up -d
\`\`\`

#### Manual Startup

1. Start the Django backend:
\`\`\`bash
cd django_backend
python manage.py runserver
\`\`\`

2. Start the FastAPI backend:
\`\`\`bash
cd fastapi_backend
uvicorn main:app --reload
\`\`\`

3. Run the data collection pipeline:
\`\`\`bash
cd data_pipeline
python -m scrapers.match_scraper
\`\`\`

4. Train the ML models:
\`\`\`bash
cd ml
python -m training.train
\`\`\`

## API Documentation

- Django API: http://localhost:8000/api/docs/
- FastAPI: http://localhost:8001/docs

## Testing

Run the test suite:

\`\`\`bash
# Django tests
cd django_backend
python manage.py test

# FastAPI tests
cd fastapi_backend
pytest

# ML model tests
cd ml
pytest
\`\`\`

## Project Structure

- `ml/`: Machine learning models and training pipelines
- `django_backend/`: Django REST Framework backend
- `fastapi_backend/`: FastAPI backend implementation
- `data_pipeline/`: Web scraping and data processing

