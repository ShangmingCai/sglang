// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-request plan for the observational tees: the ingress ids to the
//! theoretical cache-sim / token export once the request reached a worker,
//! and the completed reply captured off the response for the extend tee.
//!
//! Built from the prepared request BEFORE dispatch (the prepared request is
//! consumed by it), fired AFTER dispatch returned, so a request shed at
//! admission or selection is never counted: it never reached an engine, so
//! counting it would inflate the oracle's denominator exactly when the fleet
//! is shedding. Nothing here can slow, block or fail the serving path.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{HeaderMap, Response};
use bytes::Bytes;

use super::preparation::PreparedRequest;
use super::Endpoint;
use crate::server::app_context::AppContext;
use crate::server::cache_sim_extend::{self, ReplySource};
use crate::server::cache_sim_tee::Attribution;

pub(super) struct TeePlan {
    model: String,
    request_id: String,
    ids: Option<Vec<u32>>,
    body: Bytes,
    streaming: bool,
    extend_armed: bool,
    attr: Attribution,
}

impl TeePlan {
    /// `None` when neither tee is configured, so the hot path pays nothing.
    pub(super) fn new(
        ctx: &AppContext,
        endpoint: Endpoint,
        request: &PreparedRequest,
        headers: &HeaderMap,
    ) -> Option<Self> {
        if ctx.cache_sim_tee.is_none() && ctx.s3_export_sink.is_none() {
            return None;
        }
        let ids = request.tokens.as_ref().map(|t| t.ids.clone());
        // The extension only makes sense for chat: it re-sends the reply as
        // history next round. It also needs this round's ids (a request that
        // could not tokenize now cannot be matched next round either) and a
        // body shape the next round renders identically.
        let extend_armed = matches!(endpoint, Endpoint::Chat)
            && ids.is_some()
            && serde_json::from_slice::<serde_json::Value>(&request.body)
                .ok()
                .is_some_and(|v| cache_sim_extend::extension_can_match(&v));
        Some(Self {
            model: request.model.0.clone(),
            request_id: derive_request_id(headers),
            ids,
            body: request.body.clone(),
            streaming: request.streaming,
            extend_armed,
            attr: tee_attribution(headers),
        })
    }

    /// Fire the ingress tee for a request that reached a worker, and arm the
    /// response capture for the extend tee on a successful chat response.
    pub(super) fn after_dispatch(self, ctx: &Arc<AppContext>, response: Response<Body>) -> Response<Body> {
        if let Some(ids) = self.ids.as_ref() {
            if let Some(tee) = ctx.cache_sim_tee.as_ref() {
                tee.offer(&self.model, ids, &self.request_id, self.attr.clone());
            }
            if let Some(sink) = ctx.s3_export_sink.as_ref() {
                sink.offer_ingest(
                    &self.model,
                    ids,
                    &self.request_id,
                    self.attr.slug.as_deref(),
                    self.attr.key_id.as_deref(),
                );
            }
        }
        if !self.extend_armed || !response.status().is_success() {
            return response;
        }
        // Reserve a global capture slot so aggregate capture memory stays
        // bounded (N x MAX_EXTEND_CAPTURE_BYTES). Budget exhausted: skip this
        // response's extend tee, counted as `capture_capped`.
        let permit = if let Some(tee) = ctx.cache_sim_tee.as_ref() {
            let permit = tee.try_acquire_capture_permit();
            if permit.is_none() {
                ctx.metrics.record_cache_sim_tee("capture_capped");
                if ctx.s3_export_sink.is_some() {
                    ctx.metrics.record_s3_export("dropped_capture_capped");
                }
                return response;
            }
            permit
        } else if let Some(sink) = ctx.s3_export_sink.as_ref() {
            let permit = sink.try_acquire_capture_permit();
            if permit.is_none() {
                ctx.metrics.record_s3_export("dropped_capture_capped");
                return response;
            }
            permit
        } else {
            return response;
        };
        let Some(permit) = permit else {
            return response;
        };
        let ctx = Arc::clone(ctx);
        let TeePlan {
            model,
            request_id,
            body,
            streaming,
            attr,
            ..
        } = self;
        cache_sim_extend::capture_response_body(
            response,
            cache_sim_extend::MAX_EXTEND_CAPTURE_BYTES,
            permit,
            Box::new(move |buf| {
                let source = if streaming {
                    ReplySource::Sse(buf)
                } else {
                    ReplySource::Json(Bytes::from(buf))
                };
                cache_sim_extend::spawn_extend_tee(
                    ctx,
                    model,
                    request_id,
                    body,
                    source,
                    attr.slug,
                    attr.key_id,
                );
            }),
        )
    }
}

/// The join key between a request's ingest record and its extend record. An
/// upstream `x-request-id` is reused (prefixed, so it cannot collide with a
/// caller's own `rid` namespace); otherwise a fresh id is minted. Both tees
/// get the same value from the same plan, so they can never disagree.
fn derive_request_id(headers: &HeaderMap) -> String {
    match nonempty(headers, "x-request-id") {
        Some(id) => format!("router-{id}"),
        None => format!("router-{}", uuid::Uuid::new_v4().simple()),
    }
}

/// The upstream gateway's per-request attribution headers, relayed verbatim
/// to the cache-sim so each record is self-describing. The router mints none
/// of these; absent or empty headers stay absent.
fn tee_attribution(headers: &HeaderMap) -> Attribution {
    Attribution {
        correlation_id: nonempty(headers, "x-radixark-correlation-id"),
        endpoint_id: nonempty(headers, "x-radixark-endpoint-id"),
        key_id: nonempty(headers, "x-radixark-key-id"),
        slug: nonempty(headers, "x-radixark-endpoint-slug"),
    }
}

fn nonempty(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}
