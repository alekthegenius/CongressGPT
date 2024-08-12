# Chooses Python 3.12.3 on Debian Bookworm
FROM python:3.12.3-bookworm

# Stops Python from generating .pyc file
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for better logging
ENV PYTHONUNBUFFERED=1

# Install APT Packages
RUN apt update
#RUN apt install -y

# Cleanup APT Packages
RUN apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

CMD ["streamlit", "run", "main.py"]