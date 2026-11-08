# FruitMatch CleanRL recommender

This uv project runs a local FastAPI service. The Laravel app sends each like or pass to the feedback endpoint. Python saves each response in a per-session user preferences JSON file, retrains a PPO policy, and uses that policy to pick the next available fruit.

The trainer adapts CleanRL's single-file PPO implementation for a one-step fruit preference environment. Each profile has a separate JSON preference file and policy checkpoint. User choices are not shared between browser sessions.

## Setup

From this folder, run:

    uv sync --python 3.10
    uv run fruitmatch-ml serve

The API listens on http://127.0.0.1:8001. To start the same service through the original launcher, run:

    python reinforceML.py

The Laravel project should set CLEANRL_SERVICE_URL=http://127.0.0.1:8001.

The service exposes GET /health, POST /recommend, POST /feedback, and POST /train/{profile_id}. Feedback is saved before training so a failed PPO update does not lose a like or pass.

The recommendation endpoint returns a next-fruit probability for every candidate as well as a preference estimate. Fruit names outside the initial seed catalog are supported through hashed name features, so newly added catalog items can be scored from their name and nutrition.
