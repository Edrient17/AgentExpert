# 1. Base image
FROM python:3.11-slim

# 2. Install fonts
RUN apt-get update && apt-get install -y fonts-nanum*

# 3. Set working directory
WORKDIR /app

# 4. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project files
COPY . .

# 6. Run server
CMD ["python", "api.py"]
