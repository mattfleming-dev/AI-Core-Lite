# Lite RAG

Lite RAG is a small local retrieval-augmented-generation proxy for OpenAI-compatible model servers. It adds document retrieval to chat requests and can translate Anthropic Messages requests to the upstream OpenAI-compatible API. Python's standard library is used by default; `numpy` and `pypdf` are optional.

## Run

```powershell
python -m lite_rag serve --upstream http://127.0.0.1:8080
```

Open `http://127.0.0.1:8100/` to use the local UI. To use a configuration file, copy `config.example.json` to a private local file and pass it with `--config`.

Other commands include:

```text
python -m lite_rag ingest <collection> <path...>
python -m lite_rag search <collection> "<query>"
python -m lite_rag collections
python -m lite_rag delete <collection>
```

## API endpoints

| API | Base URL | Endpoint |
|---|---|---|
| OpenAI-compatible | `http://127.0.0.1:8100/v1` | `/chat/completions` |
| Anthropic Messages | `http://127.0.0.1:8100` | `/v1/messages` |

Anthropic requests are translated for the upstream server. When supplied, `x-api-key` is forwarded upstream. Other endpoints, such as `/v1/models` and `/v1/embeddings`, pass through unchanged.

Images pass through unchanged. Retrieval uses text from the last user message; image content is preserved for vision-capable models.

## Retrieval controls

- JSON body: `"rag": false` to disable retrieval, or `"rag": {"collection": "name", "top_k": 3}` to select a collection.
- Headers: `X-RAG-Collection: a,b` (or `*` for all collections) and `X-RAG-Disable: 1`.
- Responses include `X-RAG-Sources` with the passages used.

Without an embedding model, the built-in hashing embedder is used with keyword/BM25 search. To use semantic embeddings, configure `embed.model` to an embedding model served by the upstream. Re-index a collection after changing embedders.

## Security and data

The admin routes (`/rag/*`) are intended for loopback use. If you bind to another interface, configure `admin_key` and restrict network access with appropriate firewall controls. Keep the API bound to `127.0.0.1` unless remote access is specifically needed.

Indexed documents and database files can contain private information. They are local runtime data and are not included in this repository. Do not publish API keys, indexed databases, or source documents.

## Windows installer

With PowerShell 7 and Inno Setup available:

```powershell
pwsh installer\build-installer.ps1
```

The installer build writes generated files under `build\` and `dist\`; the compiled installer is not committed.

## Tests

```powershell
python -m unittest discover -s tests
```

## License

This project is licensed under the [MIT License](LICENSE).
