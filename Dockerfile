FROM ubuntu:22.04

ARG PYSPARK_VERSION=3.4.0
# ARG JOHNSNOWLABS_VERSION=6.4.0
ARG SPARK_NLP_VERSION=6.4.2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y                                      \
    && apt-get install -y                                  \
    openjdk-8-jdk-headless                                 \
    python3-pip                                            \
    python3.10-dev

ENV HOME=/app                                                         \
    PYTHONDONTWRITEBYTECODE=1                                         \
    JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/                      \
    PATH=/usr/lib/jvm/java-8-openjdk-amd64/bin:/app/.local/bin:$PATH  \
    TF_CPP_MIN_LOG_LEVEL=2                                            \
    PYTHONIOENCODING=utf-8                                            \
    PYTHONPATH=${PYTHONPATH}:/                                        \
    HF_HOME=/opt/.prop                                                \
    PYSPARK_PYTHON=python3                                            \
    PYSPARK_DRIVER_PYTHON=python3

COPY requirements.txt /tmp/requirements.txt
WORKDIR /app

RUN pip3 install --no-cache-dir pyspark==${PYSPARK_VERSION} \
    && pip3 install --no-cache-dir --no-deps spark-nlp==${SPARK_NLP_VERSION}

# The JSL secret's first dash-delimited field is the spark-nlp-jsl version to install
# (e.g. "6.4.1-abc123..." -> 6.4.1), and the secret doubles as the index credential.
RUN --mount=type=secret,id=spark_nlp_secret \
    sh -c ' \
        SPARK_NLP_SECRET="$(cat /run/secrets/spark_nlp_secret)" \
        && JSL_VERSION="$(echo ${SPARK_NLP_SECRET} | cut -d"-" -f1)" \
        && python3 -m pip install --upgrade pip setuptools wheel \
        && pip3 install --no-cache-dir -r /tmp/requirements.txt \
        && pip3 install --no-cache-dir spark-nlp-jsl=="${JSL_VERSION}" \
           --extra-index-url "https://pypi.johnsnowlabs.com/${SPARK_NLP_SECRET}" \
    '

RUN rm -f /usr/bin/python3                                                                \
    && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* /usr/share/man/* /usr/share/doc/*    \
    && ln -s /usr/bin/python3.10 /usr/bin/python3                                         \
    && apt-get purge --remove linux-libc-dev -y                                           \
    && apt-get autoremove -y                                                              \
    && apt-get clean

COPY health_check.py /opt/health_check.py
COPY app /app

CMD ["python3", "-m", "uvicorn", "main:app", "--workers", "1", "--host", "0.0.0.0", "--port", "8510"]

# start-period covers JVM boot + the ~800MB model load on a cold cache.
HEALTHCHECK --interval=60s --start-period=900s --timeout=30s --retries=5 CMD python3 /opt/health_check.py
