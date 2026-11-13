# Mistral Chat Completion – Minimalbeispiel

Minimale Chat-Completion mit dem offiziellen Mistral Python SDK und dem Modell `mistral-small-latest`.

Dokumentation: [Chat Completions](https://docs.mistral.ai/studio/conversations/chat-completion)

## Setup

### 1. SDK installieren

```bash
pip install mistralai
```

### 2. API-Schlüssel generieren

1. Melde dich in [Mistral Studio](https://console.mistral.ai/) an (Konto erforderlich).
2. Öffne **API Keys** im Menü links.
3. Klicke auf **Create new key**, gib einen Namen ein und bestätige.
4. Kopiere den Schlüssel sofort – er wird nur einmal angezeigt.

### 3. `MISTRAL_API_KEY` im Terminal setzen

```bash
export MISTRAL_API_KEY="dein-api-schlüssel"
```

Für dauerhaftes Setzen den Befehl in `~/.bashrc` oder `~/.zshrc` eintragen.

## Ausführen

Aus dem Projektordner:

```bash
python main.py
```

Ohne gesetzten Schlüssel bricht das Skript mit einem `KeyError` ab.
