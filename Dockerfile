FROM python:3.10-slim

# Avoid Python writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

COPY config ./config 

# Expose the port Hugging Face expects (7860)
EXPOSE 7860

# Run your Flask app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "router:app"]
