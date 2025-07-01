# Dockerfile (Final Version with Symlink)

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create a non-root user and group
RUN groupadd --system app && useradd --system --gid app app

# Set the working directory
WORKDIR /app

# Install dependencies as root first to leverage Docker cache
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and set ownership
COPY --chown=app:app ./src ./src

# --- NEW SECTION: CREATE SYMBOLIC LINK ---
# This is the "brute force" fix.
# It creates the exact directory path the API is looking for.
RUN mkdir -p /home/fentahun/10_acadamy/credit-risk-model-week-5
# It then creates a symbolic link from that path to the real artifact location.
RUN ln -s /app/mlruns /home/fentahun/10_acadamy/credit-risk-model-week-5/mlruns
# --- END OF NEW SECTION ---

# Switch to the non-root user
USER app

# Expose the port
EXPOSE 8000

# Run the application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "src.api.main:app"]