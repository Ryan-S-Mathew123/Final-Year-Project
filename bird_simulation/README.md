# FlockSense

A zero-dependency multi-bird acoustic-localization simulation. Run it with:

```powershell
node server.js
```

Then open `http://localhost:4173`.

## Configuration model and API

The browser holds the centralized `DEFAULT_CONFIG` model in `app.js`, validates edits before saving them to local storage, and marks settings that need a reset. The optional local backend exposes the persisted model for integrations:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/config` | Read the effective configuration |
| `PUT` | `/api/config` | Validate and persist a complete or partial configuration |

`PUT` writes `simulation-config.json` on success. Its probability validation requires non-negative values and at least one positive weight. The weighted selector normalizes weights by their sum, so entries do not need to total 100.

## Event handling

Sound events are given a unique event ID and scheduled independently per bird. Each event builds its own device-observation collection. `solvePosition()` receives only the event observations (device position, noisy range, timestamp, and quality); it does not receive a bird ID or ground-truth coordinate. Events are localized separately, and their result is associated back to the emitting bird only by the simulation layer. Ghost markers use a separately stored raw estimate and display position.
