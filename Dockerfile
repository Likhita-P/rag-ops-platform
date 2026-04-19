FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY app/           ${LAMBDA_TASK_ROOT}/app/
COPY agent/         ${LAMBDA_TASK_ROOT}/agent/
COPY pipelines/     ${LAMBDA_TASK_ROOT}/pipelines/
COPY monitoring/    ${LAMBDA_TASK_ROOT}/monitoring/
COPY evals/         ${LAMBDA_TASK_ROOT}/evals/
COPY prompts/       ${LAMBDA_TASK_ROOT}/prompts/

# Lambda handler — points to Mangum wrapper in main.py
CMD ["app.main.handler"]
