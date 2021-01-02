# Makefile for setting up the environment
# and starting both applications

setup:
    # Create conda environment and install required packages
	conda create --name plant_detection --file requirements.txt

run_dev:
	uvicorn api:api --reload &
	cd client && npm run dev