# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /workspace

# Install Node.js and npm
RUN apt-get update && apt-get install -y nodejs npm

# Copy only the requirements file initially (to leverage Docker layer caching)
COPY requirements.txt /workspace/requirements.txt

# Copy all files in the project directory to the container’s working directory
COPY . /workspace