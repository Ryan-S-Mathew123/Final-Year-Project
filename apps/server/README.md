# Bird Localization — Final Demo Server

## Start
docker compose up --build

Open on the laptop:
http://localhost:8000

`0.0.0.0` is the server bind address. It means the server listens on all network interfaces inside the container. It is not normally a browser destination.

To connect phones, use the laptop's LAN IPv4 address:
ws://YOUR_LAPTOP_IP:8000/ws

Find it on Windows with:
ipconfig

## Dashboard
You can manually add phones even before connecting them, move them by dragging, click/select them and type exact metre coordinates, and delete disconnected phones.

At least three connected phone audio streams are needed for 2D localization.

## Important
This remains a TDOA prototype. Independent phone clocks/streams must eventually be calibrated or synchronized for meaningful real-world accuracy.
