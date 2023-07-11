FROM ubuntu:latest

WORKDIR /app

COPY . /app

WORKDIR /app/Softwaredemonstrator

RUN apt-get update && apt-get install -y \
    libxrender1 \
    xvfb \
    libglu1 \
    freeglut3-dev \
    libgl1 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    python3 \
    python3-pip \
    nano

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

EXPOSE 5000

CMD [ "python3", "app.py" ]



