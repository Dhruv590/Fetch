FROM python:3.8-slim

# Set environment variables to prevent Python from writing .pyc files and to enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port for Jupyter Notebook (optional)
EXPOSE 8888

# Default command to run Jupyter Notebook (you can override this when running the container)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]