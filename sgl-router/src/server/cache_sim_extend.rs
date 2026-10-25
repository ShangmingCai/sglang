// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Response-completion half of the cache-sim / token-export tees.
//!
//! The ingress tee ([`crate::server::cache_sim_tee`]) sends each request's
//! prompt ids to the theoretical cache-sim as a hit-rate sample. On its own
//! that under-counts: the NEXT round of a conversation re-sends this round's
//! assistant reply as history, and a real engine serves those tokens from the
//! KV cache it filled while generating them. This module reconstructs the
//! assistant reply from the completed response (JSON body or the captured SSE
//! stream), re-renders the conversation with that reply appended, and tees
//! the full prompt+reply token sequence insert-only (`/extend_ids`), plus the
//! matching `extend` record to the S3 token export.
//!
//! Port note: the fork branch (`combine/router-admission-http2-loadaware`)
//! extended the ingress ids incrementally with the fork's own DeepSeek
//! renderer. This router renders chat through `sglang-processor` /
//! dynamo-render, which exposes no incremental turn encoder, so the extension
//! is always a full re-render of `messages + reply` through
//! [`crate::tokenizer::TokenizerRegistry::encode_chat`] (the same path the
//! ingress tokenization used). Cost: one chat render per completed response,
//! off the serving path on a spawned task. No prompt/output boundary is sent
//! (the fork sent one only on its incremental path), so the receiver treats
//! the sequence as insert-only.
//!
//! Everything here is observational: a failure logs at debug and the
//! response the client sees is untouched.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::body::Body;
use axum::http::Response;
use bytes::Bytes;
use futures::Stream;
use serde_json::Value;

use crate::discovery::ModelId;
use crate::policies::request_tokens_for;
use crate::server::app_context::AppContext;

/// Upper bound on bytes buffered per response for reply reconstruction. A
/// response past this is not teed (its capture is dropped, counted nowhere:
/// it is a size guard, not an error).
pub(crate) const MAX_EXTEND_CAPTURE_BYTES: usize = 16 << 20;

/// Where the completed reply came from.
pub(crate) enum ReplySource {
    /// A buffered (non-streaming) OpenAI chat-completion JSON body.
    Json(Bytes),
    /// The raw SSE bytes of a streaming chat completion.
    Sse(Vec<u8>),
}

/// Whether a request's extension could ever match a later round's prompt.
///
/// Mirrors the fork's predicate: shapes the engine renders divergently on the
/// next round (assistant continuation, `task`, legacy per-message keys) are
/// skipped, and so is DeepSeek thinking mode without tools, where the
/// template re-renders history without the reasoning, so the extension could
/// only ever be dead blocks.
pub(crate) fn extension_can_match(request: &Value) -> bool {
    if request_has_extension_unsafe_shape(request) {
        return false;
    }
    if !thinking_enabled(request) {
        return true;
    }
    request
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
}

/// The request's effective `thinking` chat-template kwarg: the request's
/// `chat_template_kwargs.thinking` when present, else the fleet default from
/// `SGLANG_DEFAULT_THINKING` (the same env the engines and the router's chat
/// formatter read).
fn thinking_enabled(request: &Value) -> bool {
    if let Some(v) = request
        .get("chat_template_kwargs")
        .and_then(|k| k.get("thinking"))
        .filter(|v| !v.is_null())
    {
        return openai_bool(v) != Some(false);
    }
    std::env::var("SGLANG_DEFAULT_THINKING")
        .is_ok_and(|v| ["true", "1", "yes", "y"].contains(&v.to_lowercase().as_str()))
}

/// Lenient OpenAI-style boolean: `true`/`false`, their string forms, or 0/1.
fn openai_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_i64().map(|i| i != 0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" => Some(true),
            "false" | "0" | "no" | "n" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn request_has_extension_unsafe_shape(request: &Value) -> bool {
    if request
        .get("continue_final_message")
        .filter(|v| !v.is_null())
        .is_some_and(|v| openai_bool(v) != Some(false))
    {
        return true;
    }
    if request.get("task").is_some_and(|v| !v.is_null()) {
        return true;
    }
    let Some(msgs) = request.get("messages").and_then(Value::as_array) else {
        return false;
    };
    if msgs
        .last()
        .is_some_and(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
    {
        return true;
    }
    msgs.iter().any(|m| {
        ["task", "wo_eos", "mask", "content_blocks"]
            .iter()
            .any(|k| m.get(k).is_some_and(|v| !v.is_null()))
    })
}

/// Reconstruct the assistant reply from `source`, re-render the conversation
/// with it appended, and tee the result to the cache-sim (`/extend_ids`) and
/// the S3 export. Spawned; never blocks the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_extend_tee(
    ctx: Arc<AppContext>,
    model: String,
    request_id: String,
    request_body: Bytes,
    source: ReplySource,
    slug: Option<String>,
    key_id: Option<String>,
) {
    if ctx.cache_sim_tee.is_none() && ctx.s3_export_sink.is_none() {
        return;
    }
    tokio::spawn(async move {
        let replies = match &source {
            ReplySource::Json(body) => assistant_messages_from_response_json(body),
            ReplySource::Sse(bytes) => assistant_messages_from_sse(bytes),
        };
        let output_tokens = match &source {
            ReplySource::Json(body) => completion_tokens_from_response_json(body),
            ReplySource::Sse(bytes) => completion_tokens_from_sse(bytes),
        };
        if replies.is_empty() {
            tracing::debug!(model = %model, "cache-sim extend: no assistant reply reconstructed; skipping");
            return;
        }
        let model_id = ModelId(model.clone());
        let mut parsed: Option<Value> = None;
        let requested_n = serde_json::from_slice::<Value>(&request_body)
            .ok()
            .and_then(|v| v.get("n").and_then(Value::as_u64))
            .unwrap_or(1)
            .max(1) as usize;
        let choice_count = requested_n.max(replies.len());
        for (choice_index, reply) in replies {
            // The render is CPU-bound (a full chat render of the history);
            // keep it off the async workers.
            let ctx_for_render = Arc::clone(&ctx);
            let model_for_render = model_id.clone();
            let body_for_render = request_body.clone();
            let mut parsed_taken = parsed.take();
            let rendered = tokio::task::spawn_blocking(move || {
                let ids = full_reencode_extension(
                    &ctx_for_render,
                    &model_for_render,
                    &body_for_render,
                    &mut parsed_taken,
                    reply,
                );
                (ids, parsed_taken)
            })
            .await;
            let ids = match rendered {
                Ok((ids, parsed_back)) => {
                    parsed = parsed_back;
                    ids
                }
                Err(_) => None,
            };
            let Some(ids) = ids else {
                tracing::debug!(model = %model, "cache-sim extend: tokenize failed; skipping");
                continue;
            };
            let choice = (choice_count > 1).then_some(crate::server::cache_sim_tee::Choice {
                index: choice_index as usize,
                count: choice_count,
            });
            let per_choice_output = if choice_count > 1 {
                None
            } else {
                output_tokens
            };
            if let Some(tee) = ctx.cache_sim_tee.as_ref() {
                tee.offer_extend(&model, &ids, &request_id, None, choice, per_choice_output);
            }
            if let Some(sink) = ctx.s3_export_sink.as_ref() {
                sink.offer_extend(
                    &model,
                    &ids,
                    &request_id,
                    None,
                    per_choice_output,
                    choice.as_ref().map(|c| c.index),
                    choice.as_ref().map(|c| c.count),
                    slug.as_deref(),
                    key_id.as_deref(),
                );
            }
        }
    });
}

/// `messages + reply` rendered through the model's chat formatter, falling
/// back to the routing tokenization of the same body. `parsed` caches the
/// parsed request across choices; the appended reply is popped again so the
/// next choice starts from the original history.
fn full_reencode_extension(
    ctx: &AppContext,
    model_id: &ModelId,
    request_body: &Bytes,
    parsed: &mut Option<Value>,
    reply: Value,
) -> Option<Vec<u32>> {
    if parsed.is_none() {
        let request = serde_json::from_slice::<Value>(request_body).ok()?;
        request.get("messages").filter(|m| m.is_array())?;
        *parsed = Some(request);
    }
    let request = parsed.as_mut()?;
    request
        .get_mut("messages")
        .and_then(Value::as_array_mut)?
        .push(reply);
    let tokens = if ctx.tokenizers.has_chat_formatter(&model_id.0) {
        ctx.tokenizers.encode_chat(&model_id.0, request)
    } else {
        None
    };
    let tokens = match tokens {
        Some(ids) => Some(ids),
        None => request_tokens_for(&ctx.tokenizers, model_id, request).map(|t| t.ids),
    };
    if let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) {
        messages.pop();
    }
    tokens
}

/// Wrap `response`'s body so its bytes are copied (up to `max_bytes`) while
/// they flow to the client, and `on_done` runs with the copy once the body
/// ends cleanly. A body that overflows `max_bytes`, errors, or is dropped
/// mid-stream (client disconnect) never fires `on_done`; `permit` is released
/// when the wrapper is dropped. The client's bytes, headers and status are
/// untouched.
pub(crate) fn capture_response_body(
    response: Response<Body>,
    max_bytes: usize,
    permit: tokio::sync::OwnedSemaphorePermit,
    on_done: Box<dyn FnOnce(Vec<u8>) + Send + 'static>,
) -> Response<Body> {
    let (parts, body) = response.into_parts();
    let captured = CaptureStream {
        inner: body.into_data_stream(),
        buf: Vec::new(),
        max_bytes,
        overflowed: false,
        on_done: Some(on_done),
        _permit: permit,
    };
    Response::from_parts(parts, Body::from_stream(captured))
}

struct CaptureStream {
    inner: axum::body::BodyDataStream,
    buf: Vec<u8>,
    max_bytes: usize,
    overflowed: bool,
    on_done: Option<Box<dyn FnOnce(Vec<u8>) + Send + 'static>>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl Stream for CaptureStream {
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let next = Pin::new(&mut self.inner).poll_next(cx);
        match &next {
            Poll::Ready(Some(Ok(chunk))) => {
                if !self.overflowed {
                    if self.buf.len() + chunk.len() > self.max_bytes {
                        self.overflowed = true;
                        self.buf = Vec::new();
                    } else {
                        self.buf.extend_from_slice(chunk);
                    }
                }
            }
            Poll::Ready(Some(Err(_))) => {
                // An upstream error ends the body without a reconstructable
                // reply; drop the callback so a later `None` cannot fire it.
                self.on_done = None;
            }
            Poll::Ready(None) => {
                if !self.overflowed {
                    if let Some(on_done) = self.on_done.take() {
                        let buf = std::mem::take(&mut self.buf);
                        on_done(buf);
                    }
                }
            }
            Poll::Pending => {}
        }
        next
    }
}

// ---- Reply reconstruction (carried over verbatim from the fork) ------------
pub(crate) fn completion_tokens_from_response_json(body: &[u8]) -> Option<u64> {
    serde_json::from_slice::<Value>(body)
        .ok()?
        .get("usage")?
        .get("completion_tokens")?
        .as_u64()
}

/// `usage.completion_tokens` from a captured SSE stream — the RESPONSE-WIDE
/// total, or `None`.
///
/// Accepts a chunk only when its `choices` is absent or empty, which is what
/// distinguishes the aggregate usage event from a per-choice one. Two client
/// options put `usage` on the wire and they mean different things:
///
/// - `stream_options.include_usage` emits one final aggregate chunk with
///   `choices: []` — the response total, which is what we want.
/// - `stream_options.continuous_usage_stats` stamps `usage` on EVERY content
///   chunk, scoped to that chunk's single choice, and can be set WITHOUT
///   `include_usage` (so no aggregate ever arrives).
///
/// Taking the last `usage` seen would therefore return one choice's partial
/// count on a continuous-stats stream and present it as the response total —
/// a number that looks authoritative and is silently short. Absent is the
/// honest answer there.
pub(crate) fn completion_tokens_from_sse(bytes: &[u8]) -> Option<u64> {
    let text = String::from_utf8_lossy(bytes);
    for line in text.split('\n').rev() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let Some(payload) = line.strip_prefix("data:").map(str::trim_start) else {
            continue;
        };
        if payload == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        // Per-choice usage rides alongside content; only the aggregate event
        // carries no choices.
        let is_aggregate = v
            .get("choices")
            .is_none_or(|c| c.as_array().is_some_and(|a| a.is_empty()));
        if !is_aggregate {
            continue;
        }
        if let Some(n) = v
            .get("usage")
            .and_then(|u| u.get("completion_tokens"))
            .and_then(Value::as_u64)
        {
            return Some(n);
        }
    }
    None
}

/// Pull the assistant message(s) out of a buffered (non-streaming) chat
/// completion response: `choices[*].message`, verbatim — the exact object a
/// client would echo back as history next round (the chat encoders ignore
/// fields they don't render, e.g. `reasoning_content`).
pub(crate) fn assistant_messages_from_response_json(body: &[u8]) -> Vec<(u64, Value)> {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return Vec::new();
    };
    let Some(choices) = v.get("choices").and_then(Value::as_array) else {
        return Vec::new();
    };
    choices
        .iter()
        .filter_map(|c| {
            let msg = c.get("message").filter(|m| m.is_object())?.clone();
            // The ENGINE's ordinal, not this vector's position. A choice the
            // engine emitted but we could not reconstruct is skipped, so a
            // positional index would silently renumber every later choice and
            // point accounting at the wrong alternative.
            let idx = c.get("index").and_then(Value::as_u64).unwrap_or(0);
            Some((idx, msg))
        })
        .collect()
}

/// Per-choice accumulator for streamed deltas.
#[derive(Default)]
struct ChoiceAcc {
    content: String,
    /// Thinking-mode reasoning deltas. Carried on the rebuilt message so the
    /// chat encoder can render it where the next round's history rendering
    /// would (DSV4 keeps prior-turn reasoning in tool-carrying thinking
    /// conversations — the agentic case); chat-mode rendering ignores it.
    reasoning_content: String,
    /// keyed by the delta's tool-call `index`; ordered map so the rebuilt
    /// `tool_calls` array preserves the engine's emission order.
    tool_calls: std::collections::BTreeMap<u64, ToolCallAcc>,
}

#[derive(Default)]
struct ToolCallAcc {
    id: Option<String>,
    name: String,
    arguments: String,
}

/// Reconstruct the assistant message(s) from a captured SSE stream by
/// replaying the OpenAI chunk deltas: `choices[*].delta.content` and
/// `choices[*].delta.reasoning_content` concatenate (into separate fields —
/// the chat encoder decides whether reasoning renders into history);
/// `choices[*].delta.tool_calls[*]` merge by tool-call `index` (`id`/`name`
/// arrive once, `arguments` fragments concatenate). Returns one message per
/// choice index, in index order.
///
/// The input is the FULL byte stream (the pump only delivers cleanly-completed
/// captures), so SSE events split across network chunks reassemble here via
/// plain line splitting. Unparsable `data:` payloads are skipped — one
/// malformed event costs its delta, not the reconstruction.
pub(crate) fn assistant_messages_from_sse(bytes: &[u8]) -> Vec<(u64, Value)> {
    let mut choices: std::collections::BTreeMap<u64, ChoiceAcc> = std::collections::BTreeMap::new();
    let text = String::from_utf8_lossy(bytes);
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let Some(payload) = line.strip_prefix("data:").map(str::trim_start) else {
            continue; // event:/comment/blank separator lines
        };
        if payload == "[DONE]" || payload.is_empty() {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        let Some(chunk_choices) = chunk.get("choices").and_then(Value::as_array) else {
            continue;
        };
        for c in chunk_choices {
            let idx = c.get("index").and_then(Value::as_u64).unwrap_or(0);
            let acc = choices.entry(idx).or_default();
            let Some(delta) = c.get("delta") else {
                continue;
            };
            if let Some(s) = delta.get("content").and_then(Value::as_str) {
                acc.content.push_str(s);
            }
            if let Some(s) = delta.get("reasoning_content").and_then(Value::as_str) {
                acc.reasoning_content.push_str(s);
            }
            let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) else {
                continue;
            };
            for tc in tcs {
                let tc_idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
                let tacc = acc.tool_calls.entry(tc_idx).or_default();
                if let Some(id) = tc.get("id").and_then(Value::as_str) {
                    tacc.id = Some(id.to_string());
                }
                if let Some(f) = tc.get("function") {
                    if let Some(name) = f.get("name").and_then(Value::as_str) {
                        tacc.name.push_str(name);
                    }
                    if let Some(args) = f.get("arguments").and_then(Value::as_str) {
                        tacc.arguments.push_str(args);
                    }
                }
            }
        }
    }

    choices
        .into_iter()
        .filter_map(|(idx, acc)| {
            if acc.content.is_empty()
                && acc.reasoning_content.is_empty()
                && acc.tool_calls.is_empty()
            {
                return None; // nothing generated (e.g. an aborted empty stream)
            }
            let mut msg = serde_json::json!({
                "role": "assistant",
                "content": acc.content,
            });
            if !acc.reasoning_content.is_empty() {
                msg["reasoning_content"] = Value::String(acc.reasoning_content);
            }
            if !acc.tool_calls.is_empty() {
                let tool_calls: Vec<Value> = acc
                    .tool_calls
                    .into_values()
                    .map(|t| {
                        serde_json::json!({
                            "id": t.id.unwrap_or_default(),
                            "type": "function",
                            "function": {"name": t.name, "arguments": t.arguments},
                        })
                    })
                    .collect();
                msg["tool_calls"] = Value::Array(tool_calls);
            }
            Some((idx, msg))
        })
        .collect()
}

