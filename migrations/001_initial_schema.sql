-- Vanara.ai Supabase schema — v1
-- Run this in Supabase SQL Editor after creating the project.
-- Tables are RLS-protected: users can only read/write their own rows.

-- ══════════════════════════════════════════════════════════════════════
-- Extensions
-- ══════════════════════════════════════════════════════════════════════
create extension if not exists "pgcrypto";

-- ══════════════════════════════════════════════════════════════════════
-- jobs
-- ══════════════════════════════════════════════════════════════════════
create table if not exists public.jobs (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid not null references auth.users(id) on delete cascade,
  title           text,
  company         text,
  description     text not null,
  description_hash text not null,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);
create index if not exists idx_jobs_user_id on public.jobs(user_id);
create index if not exists idx_jobs_desc_hash on public.jobs(description_hash);

-- ══════════════════════════════════════════════════════════════════════
-- parsed_resumes  (Smart Library cache)
-- ══════════════════════════════════════════════════════════════════════
create table if not exists public.parsed_resumes (
  id             uuid primary key default gen_random_uuid(),
  user_id        uuid not null references auth.users(id) on delete cascade,
  filename       text not null,
  file_hash      text not null,
  parsed_resume  jsonb not null,
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now()
);
create index if not exists idx_parsed_resumes_user_id on public.parsed_resumes(user_id);
create index if not exists idx_parsed_resumes_hash on public.parsed_resumes(user_id, file_hash);

-- ══════════════════════════════════════════════════════════════════════
-- resume_generations  (History)
-- ══════════════════════════════════════════════════════════════════════
create table if not exists public.resume_generations (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid not null references auth.users(id) on delete cascade,
  job_id             uuid not null references public.jobs(id) on delete cascade,
  parsed_resume_id   uuid references public.parsed_resumes(id) on delete set null,
  original_filename  text,
  resume_json        jsonb not null,
  ats_score          numeric,
  resume_template    text default 'resume_template_7.html',
  created_at         timestamptz not null default now()
);
create index if not exists idx_resume_generations_user_id on public.resume_generations(user_id);
create index if not exists idx_resume_generations_job_id on public.resume_generations(job_id);
create index if not exists idx_resume_generations_parsed_id on public.resume_generations(parsed_resume_id);

-- ══════════════════════════════════════════════════════════════════════
-- requests  (audit log)
-- ══════════════════════════════════════════════════════════════════════
create table if not exists public.requests (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users(id) on delete set null,
  user_email  text,
  user_name   text,
  endpoint    text not null,
  status      text,
  metadata    jsonb,
  created_at  timestamptz not null default now()
);
create index if not exists idx_requests_user_id on public.requests(user_id);
create index if not exists idx_requests_created_at on public.requests(created_at desc);

-- ══════════════════════════════════════════════════════════════════════
-- feedback
-- ══════════════════════════════════════════════════════════════════════
create table if not exists public.feedback (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users(id) on delete set null,
  user_email  text,
  user_name   text,
  category    text not null,
  message     text not null,
  created_at  timestamptz not null default now()
);
create index if not exists idx_feedback_user_id on public.feedback(user_id);

-- ══════════════════════════════════════════════════════════════════════
-- updated_at trigger (jobs + parsed_resumes)
-- ══════════════════════════════════════════════════════════════════════
create or replace function public.set_updated_at() returns trigger
  language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end $$;

drop trigger if exists trg_jobs_updated_at on public.jobs;
create trigger trg_jobs_updated_at before update on public.jobs
  for each row execute function public.set_updated_at();

drop trigger if exists trg_parsed_resumes_updated_at on public.parsed_resumes;
create trigger trg_parsed_resumes_updated_at before update on public.parsed_resumes
  for each row execute function public.set_updated_at();

-- ══════════════════════════════════════════════════════════════════════
-- Row-Level Security — users own their own data
-- ══════════════════════════════════════════════════════════════════════
alter table public.jobs               enable row level security;
alter table public.parsed_resumes     enable row level security;
alter table public.resume_generations enable row level security;
alter table public.requests           enable row level security;
alter table public.feedback           enable row level security;

-- jobs
drop policy if exists "jobs_owner_all" on public.jobs;
create policy "jobs_owner_all" on public.jobs
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- parsed_resumes
drop policy if exists "parsed_resumes_owner_all" on public.parsed_resumes;
create policy "parsed_resumes_owner_all" on public.parsed_resumes
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- resume_generations
drop policy if exists "resume_generations_owner_all" on public.resume_generations;
create policy "resume_generations_owner_all" on public.resume_generations
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- requests  (inserts via service role from backend; users can read their own)
drop policy if exists "requests_owner_read" on public.requests;
create policy "requests_owner_read" on public.requests
  for select using (auth.uid() = user_id);

-- feedback (users can insert their own + read their own)
drop policy if exists "feedback_owner_all" on public.feedback;
create policy "feedback_owner_all" on public.feedback
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- ══════════════════════════════════════════════════════════════════════
-- Done.
-- Next: Enable Google OAuth in Authentication → Providers → Google
-- Add redirect URL: http://localhost:3000/auth/callback (dev)
-- ══════════════════════════════════════════════════════════════════════
