# Support

## Before opening an issue

Check [README.md](README.md), reproduce with the smallest possible request, and note the Python version, upstream API type, command used, and first relevant error.

## Troubleshooting

- **The service does not start:** use a supported Python interpreter and check that the configured listen port is available.
- **The upstream is unreachable:** verify that the model server is running at `upstream_url` and that the URL is reachable from the same machine.
- **Retrieval returns no useful results:** confirm the collection contains indexed documents. Re-index after changing the embedding model; otherwise keyword search is used.
- **The browser UI cannot administer the server:** admin routes are intended for loopback. If listening on a non-loopback interface, configure `admin_key` and restrict network access.
- **A PDF cannot be ingested:** install the optional `pypdf` package. `numpy` is also optional for embedding-related features.
- **The Windows installer fails:** run the build from PowerShell 7 and confirm Inno Setup is installed and available to the build script.

## Privacy and safe reports

The local database may contain indexed document text and metadata. Do not attach the database, private source documents, API keys, tokens, or unredacted logs to an issue. Use synthetic sample content and remove personal paths before sharing configuration or screenshots.

## Licensing

No project license is included. Public visibility does not grant permission to reuse or redistribute this code.
