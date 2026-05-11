dev:
	uv run story_gen.py & \
	python -m http.server -d ChatBotThing -b 127.0.0.1 5500