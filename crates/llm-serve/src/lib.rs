//! thotbook-AmentI M5 — private OpenAI-compatible inference server (Qwen3-8B, candle, CPU).
//! Enhanced with RAG context retrieval and agent dispatch.

pub mod agent;
pub mod config;
pub mod engine;
pub mod error;
pub mod http;
pub mod rag;

use config::ServeConfig;
use error::ServeError;

/// Load the engine, bind the listener, and serve until the process stops.
pub fn run_server(config: ServeConfig) -> Result<(), ServeError> {
    let engine = engine::Engine::load(&config)?;
    let state = http::AppState::new(engine, config.clone());

    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!(%addr, model = %config.model_id, "llm-serve listening (private, cpu)");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| ServeError::http(format!("failed to build tokio runtime: {e}")))?;

    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| ServeError::http(format!("failed to bind {addr}: {e}")))?;
        axum::serve(listener, http::router(state))
            .await
            .map_err(|e| ServeError::Http(e.to_string()))
    })
}
