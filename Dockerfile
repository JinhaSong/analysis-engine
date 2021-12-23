FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

RUN apt-get update \
    && apt-get -y install python3 python3-pip python3-dev \
    mysql-client libmysqlclient-dev python3-mysqldb \
    git wget ssh vim locales \
    apt-utils libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV DOCKERIZE_VERSION v0.6.1
RUN wget -q https://github.com/jwilder/dockerize/releases/download/$DOCKERIZE_VERSION/dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && tar -C /usr/local/bin -xzvf dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && rm dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

RUN pip3 install --upgrade pip
RUN pip3 install setuptools

WORKDIR /workspace
ADD . .
RUN pip3 install -r requirements.txt

ENV LANGUAGE=ko_KR.UTF-8
ENV LANG=ko_KR.UTF-8
RUN locale-gen ko_KR ko_KR.UTF-8
RUN update-locale LANG=ko_KR.UTF-8
ENV PYTHONIOENCODING="UTF-8"

ENV DJANGO_SUPERUSER_USERNAME root
ENV DJANGO_SUPERUSER_EMAIL none@none.com
ENV DJANGO_SUPERUSER_PASSWORD password

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
#ENTRYPOINT ["/docker-entrypoint.sh"]

RUN chmod -R a+w /workspace

EXPOSE 8000

