-- Migration: create public.voice_transcriptions table for idempotent ASR audit logging & R2 backup references
-- Follows Supabase Postgres best practices:
--   1. SQL-standard `bigint generated always as identity` primary key (avoids random UUIDv4 B-tree fragmentation)
--   2. Unique constraint on `request_id` enabling atomic idempotent `INSERT ... ON CONFLICT (request_id) DO UPDATE`
--   3. `timestamptz` for timezone-aware timestamps, `text` instead of `varchar(n)`, `jsonb` for segments
--   4. Row Level Security (RLS) enabled with anon/authenticated revoked by default (backend writes via SUPABASE_SECRET_KEY)

create table if not exists public.voice_transcriptions (
    id bigint generated always as identity primary key,
    request_id text not null,
    session_id text null,
    source text not null check (source in ('http', 'sse', 'openai', 'websocket')),
    original_filename text not null,
    content_type text null,
    file_size_bytes bigint null check (file_size_bytes is null or file_size_bytes >= 0),
    audio_sha256 text null,
    r2_bucket text null,
    r2_object_key text null,
    r2_url text null,
    archive_error text null,
    model text not null,
    engine text not null,
    language text null,
    transcript_text text not null,
    duration_seconds double precision not null,
    processing_time_seconds double precision not null,
    rtf double precision not null,
    segments jsonb null,
    created_at timestamptz not null default now(),
    constraint voice_transcriptions_request_id_key unique (request_id)
);

comment on table public.voice_transcriptions is
    'Idempotent audit log of voice recordings backed up to Cloudflare R2 and transcribed by the Nigerian Accented English ASR API.';

-- Indexes for dashboard & session queries (`request_id` is already indexed by its UNIQUE constraint)
create index if not exists voice_transcriptions_created_at_idx
    on public.voice_transcriptions (created_at desc);

create index if not exists voice_transcriptions_session_id_idx
    on public.voice_transcriptions (session_id)
    where session_id is not null;

-- Security: always enable RLS on tables in public schema.
-- Server-side upserts use SUPABASE_SECRET_KEY (sb_secret_... or service_role), which bypasses RLS.
-- Public/client roles have no direct access unless you explicitly add scoped policies for your app.
alter table public.voice_transcriptions enable row level security;
revoke all on table public.voice_transcriptions from anon, authenticated;
