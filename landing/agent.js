/**
 * Noelle Agent & WebMCP Discovery Bridge
 * Enables in-browser AI agents and WebMCP tools to inspect Noelle features,
 * hardware advisor guidelines, local API configurations, and documentation.
 */

(function () {
  'use strict';

  var NoelleAgent = {
    version: '1.0.0',

    getInfo: function () {
      return {
        name: 'Noelle',
        tagline: 'Your local AI model lab',
        description: 'Free desktop app for macOS, Windows and Linux that finds open AI models that fit your computer, runs them locally, tests them on your own tasks and serves them to your apps. Nothing leaves your machine.',
        license: 'MIT',
        website: 'https://noelle.nevatal.id',
        source: (window.NOELLE_LINKS && window.NOELLE_LINKS.source) || 'https://github.com/thirza258/NoelleApp',
        platforms: ['macOS (Apple Silicon & Intel)', 'Windows (Nvidia CUDA, DirectML, CPU)', 'Linux (Nvidia CUDA, AMD ROCm, CPU)'],
        modelFormats: ['GGUF (llama.cpp)', 'MLX (Apple Silicon)', 'EXL2', 'Transformers'],
        localApi: 'http://127.0.0.1:8765/v1',
        mcpCommand: 'noelle mcp'
      };
    },

    getHardwareAdvisor: function (vramGB) {
      var vram = parseFloat(vramGB);
      var recommendations = [];

      if (isNaN(vram) || vram <= 4) {
        recommendations.push({
          tier: '1B - 3B Models (Q4_K_M)',
          examples: ['SmolLM2-1.7B-Instruct', 'Llama-3.2-1B-Instruct', 'Qwen2.5-1.5B-Instruct'],
          memoryRequirement: '1.5 GB - 3 GB',
          recommendedQuant: 'Q4_K_M / Q5_K_M'
        });
      }
      if (isNaN(vram) || (vram > 4 && vram <= 8)) {
        recommendations.push({
          tier: '7B - 8B Models (Q4_K_M)',
          examples: ['Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct'],
          memoryRequirement: '5.5 GB - 7 GB',
          recommendedQuant: 'Q4_K_M'
        });
      }
      if (isNaN(vram) || (vram > 8 && vram <= 16)) {
        recommendations.push({
          tier: '12B - 14B Models (Q4/Q5) or 8B (Q8_0 / BF16)',
          examples: ['Gemma-2-9B-Instruct', 'Qwen2.5-14B-Instruct', 'Llama-3.1-8B (Q8)'],
          memoryRequirement: '9 GB - 14 GB',
          recommendedQuant: 'Q4_K_M / Q8_0'
        });
      }
      if (isNaN(vram) || vram > 16) {
        recommendations.push({
          tier: '32B - 70B Quantized Models',
          examples: ['Qwen2.5-32B-Instruct (Q4_K_M)', 'Llama-3.1-70B-Instruct (IQ3/Q4)'],
          memoryRequirement: '18 GB - 48 GB',
          recommendedQuant: 'Q4_K_M / EXL2 4.0bpw'
        });
      }

      return {
        queriedVRAM: isNaN(vram) ? 'All Tiers' : vram + ' GB',
        recommendations: recommendations,
        advice: 'Noelle automatically calculates the weights, KV cache, and context overhead to give a 1-click Fit Score for any model.'
      };
    },

    getMcpConfig: function (client) {
      var target = (client || 'claude').toLowerCase();
      var baseConfig = {
        claudeCode: {
          command: 'claude mcp add noelle -- noelle mcp'
        },
        claudeDesktop: {
          mcpServers: {
            noelle: {
              command: 'noelle',
              args: ['mcp']
            }
          }
        },
        cursor: {
          name: 'noelle',
          type: 'command',
          command: 'noelle mcp'
        }
      };

      if (target.indexOf('code') !== -1) return baseConfig.claudeCode;
      if (target.indexOf('cursor') !== -1) return baseConfig.cursor;
      return baseConfig.claudeDesktop;
    },

    getApiDocs: function () {
      return {
        baseUrl: 'http://127.0.0.1:8765/v1',
        format: 'OpenAI-Compatible',
        endpoints: [
          { method: 'GET', path: '/v1/models', desc: 'List downloaded models and custom aliases' },
          { method: 'POST', path: '/v1/chat/completions', desc: 'Chat completions with SSE streaming support' },
          { method: 'POST', path: '/v1/embeddings', desc: 'Generate vectors with local embedding models' },
          { method: 'POST', path: '/v1/audio/transcriptions', desc: 'Speech to text transcription' },
          { method: 'POST', path: '/v1/images/generations', desc: 'Text to image generation (base64 b64_json)' },
          { method: 'GET', path: '/v1/knowledge', desc: 'List local knowledge bases' },
          { method: 'POST', path: '/v1/knowledge/query', desc: 'Search local knowledge base with top_k' },
          { method: 'GET', path: '/health', desc: 'Runtime and hardware health status' }
        ],
        autoRouting: 'Set "model": "auto" for Noelle to automatically pick the best loaded or suited model.'
      };
    },

    getDownloadInfo: function () {
      var links = window.NOELLE_LINKS || {};
      return {
        projectUrl: links.project || 'https://sourceforge.net/projects/noelleapp/',
        downloadUrl: links.download || 'https://sourceforge.net/projects/noelleapp/files/latest/download',
        allFilesUrl: links.files || 'https://sourceforge.net/projects/noelleapp/files/',
        sourceUrl: links.source || 'https://github.com/thirza258/NoelleApp',
        platforms: {
          macos: 'Universal (Apple Silicon M1/M2/M3/M4 & Intel DMG)',
          windows: 'x64 Installer (Nvidia CUDA & DirectML)',
          linux: 'AppImage & DEB/RPM packages'
        }
      };
    },

    searchDocs: function (query) {
      var q = (query || '').toLowerCase().trim();
      var docs = [
        { title: 'Overview', path: '/docs/', desc: 'Get started with Noelle local model lab' },
        { title: 'Installation', path: '/docs/install/', desc: 'System requirements, downloads and installation' },
        { title: 'Quick Start', path: '/docs/quickstart/', desc: 'Download your first model and run chat in 2 minutes' },
        { title: 'Models & Hardware Fit', path: '/docs/models/', desc: 'Search Hugging Face models and check hardware fit' },
        { title: 'Evaluation Lab', path: '/docs/evaluation/', desc: 'Benchmark and compare prompt responses side-by-side' },
        { title: 'Knowledge Canvas (RAG)', path: '/docs/knowledge/', desc: 'Local document indexing and semantic retrieval' },
        { title: 'Local API', path: '/docs/api/', desc: 'OpenAI-compatible endpoints and auto-routing' },
        { title: 'MCP Server', path: '/docs/mcp/', desc: 'Connect Claude Code, Cursor and external agents' },
        { title: 'CLI Reference', path: '/docs/cli/', desc: 'Command line interface for headless execution' },
        { title: 'Privacy & Offline Guarantee', path: '/docs/privacy/', desc: 'Zero telemetry, fully local execution' }
      ];

      if (!q) return docs;
      return docs.filter(function (d) {
        return d.title.toLowerCase().indexOf(q) !== -1 || d.desc.toLowerCase().indexOf(q) !== -1;
      });
    }
  };

  // Expose globally
  window.NoelleAgent = NoelleAgent;

  // WebMCP Registration
  function registerWebMCP() {
    var mc = (typeof navigator !== 'undefined' && navigator.modelContext) ||
             (typeof document !== 'undefined' && document.modelContext);
    if (!mc) return;

    var tools = [
      {
        name: 'get_noelle_info',
        description: 'Get general overview, architecture, capabilities, and download options for Noelle local AI model lab.',
        inputSchema: { type: 'object', properties: {} },
        execute: async function () {
          return NoelleAgent.getInfo();
        }
      },
      {
        name: 'get_hardware_advisor',
        description: 'Get model fit recommendations and quantization advice based on available VRAM in GB.',
        inputSchema: {
          type: 'object',
          properties: {
            vramGB: { type: 'number', description: 'Available VRAM in gigabytes (e.g. 8, 16, 24)' }
          }
        },
        execute: async function (args) {
          return NoelleAgent.getHardwareAdvisor(args && args.vramGB);
        }
      },
      {
        name: 'get_mcp_config',
        description: 'Get configuration snippet for Claude Code, Cursor, or Claude Desktop to connect to Noelle MCP.',
        inputSchema: {
          type: 'object',
          properties: {
            client: { type: 'string', description: 'Target client: claudeCode, cursor, or claudeDesktop' }
          }
        },
        execute: async function (args) {
          return NoelleAgent.getMcpConfig(args && args.client);
        }
      },
      {
        name: 'get_api_docs',
        description: 'Get documentation on Noelle local OpenAI-compatible API endpoints and automatic routing.',
        inputSchema: { type: 'object', properties: {} },
        execute: async function () {
          return NoelleAgent.getApiDocs();
        }
      },
      {
        name: 'get_download_info',
        description: 'Get download URLs and package availability for macOS, Windows, and Linux.',
        inputSchema: { type: 'object', properties: {} },
        execute: async function () {
          return NoelleAgent.getDownloadInfo();
        }
      }
    ];

    if (typeof mc.provideContext === 'function') {
      try { mc.provideContext({ tools: tools }); } catch (e) {}
    }
    if (typeof mc.registerTool === 'function') {
      tools.forEach(function (tool) {
        try { mc.registerTool(tool); } catch (e) {}
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', registerWebMCP);
  } else {
    registerWebMCP();
  }
})();
