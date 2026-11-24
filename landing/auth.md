# auth.md — Autonomous Agent Interaction & Authentication Guide

## Overview
This document guides autonomous AI agents, web crawlers, and developer tools interacting with the Noelle website (`https://noelle.nevatal.id`) and Noelle desktop instances.

## Discovery Metadata
- Agent Card (A2A standard): https://noelle.nevatal.id/.well-known/agent-card.json
- AI Catalog: https://noelle.nevatal.id/.well-known/ai-catalog.json
- API Catalog Linkset: https://noelle.nevatal.id/.well-known/api-catalog
- MCP Server Card: https://noelle.nevatal.id/.well-known/mcp/server-card.json
- Poppy Agent Protocol: https://noelle.nevatal.id/.well-known/poppy.json
- OAuth Protected Resource: https://noelle.nevatal.id/.well-known/oauth-protected-resource
- OAuth Authorization Server: https://noelle.nevatal.id/.well-known/oauth-authorization-server
- OpenID Configuration: https://noelle.nevatal.id/.well-known/openid-configuration

## Authentication Policies

### 1. Website & Documentation Access (Public)
All documentation, guides, discovery endpoints, site maps, and download links on `https://noelle.nevatal.id/` are 100% public. No authentication tokens, account registration, or cookies are required.
Agents may freely index and query this site using standard HTTP `GET` requests. Content negotiation is supported: requests with `Accept: text/markdown` at `/` receive `/llms.txt`.

### 2. Client-Side WebMCP In-Browser Agent Discovery
Browsers or AI agent extensions visiting the site can leverage WebMCP tools registered directly on `navigator.modelContext` / `document.modelContext` via `/agent.js`. Tools available include:
- `get_noelle_info`
- `get_hardware_advisor`
- `get_mcp_config`
- `get_api_docs`
- `get_download_info`

### 3. Local Desktop API (`http://127.0.0.1:8765/v1`)
When running the Noelle desktop application locally:
- The API server binds by default to loopback interface `127.0.0.1:8765`.
- By default, authentication is disabled (`api_key="not-needed"`).
- If an API key is configured by the user in Noelle Settings, supply it via `Authorization: Bearer <local-key>`.
- The API is strictly on-device; no traffic or credentials are routed through external clouds or telemetry servers.
