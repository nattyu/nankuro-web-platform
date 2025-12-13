FROM public.ecr.aws/lambda/python:3.12

# Install dependencies
COPY requirements_lambda.txt requirements.txt
# If there are separate ML requirements, install them too
# COPY requirements_nankuro.txt . 
# RUN pip install -r requirements_nankuro.txt --target "${LAMBDA_TASK_ROOT}" 
# (Assuming requirements.txt covers everything for now, or user will consolidate)

# Debug environment
RUN python --version && pip debug --verbose
RUN pip install --upgrade pip
RUN pip install setuptools wheel
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --target "${LAMBDA_TASK_ROOT}"
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}" --extra-index-url https://download.pytorch.org/whl/cpu

# FIX: Remove standard opencv-python (dependencies of ultralytics) and force headless
# to avoid libGL.so.1 error in Lambda.
# pip uninstall does not work with --target, so we must remove the files manually.
RUN rm -rf ${LAMBDA_TASK_ROOT}/cv2 ${LAMBDA_TASK_ROOT}/opencv_python*
RUN pip install opencv-python-headless --target "${LAMBDA_TASK_ROOT}" --no-deps

# Copy the entire project context (filtered by .dockerignore)
# This includes:
# - solver_core/
# - lambda_app/
# - models/
# - custom_yolo_*/
# - config.py
# - jukugo.csv
# - etc.
COPY . ${LAMBDA_TASK_ROOT}

# The CMD is relative to valid python path. 
# Since we copied everything to ROOT, lambda_app/lambda_handler.py is at ${LAMBDA_TASK_ROOT}/lambda_app/lambda_handler.py
# But we need to make sure python can find it.
# We can set CMD to "lambda_app.lambda_handler.lambda_handler" if __init__.py exists or just move it.
# Or we can adjust PYTHONPATH.
# Simpler: Stick to the previous plan but verify path.
# Previous: COPY lambda_app/lambda_handler.py ${LAMBDA_TASK_ROOT}/
# Now: It is inside lambda_app folder in root. 
# Let's override CMD to point to the module path.

CMD ["lambda_app.lambda_handler.lambda_handler"]
