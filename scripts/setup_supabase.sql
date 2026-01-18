-- =============================================================================
-- JAVIS Cloud Storage Schema for Supabase
-- =============================================================================
-- This script creates the necessary tables for JAVIS cloud storage.
-- Run this script in your Supabase SQL Editor.
-- =============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- 1. Conversations Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    feedback VARCHAR(50), -- 'good', 'bad', or NULL
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_started_at ON conversations(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_feedback ON conversations(feedback);

-- =============================================================================
-- 2. Conversation Turns Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS conversation_turns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_index INTEGER NOT NULL,
    role VARCHAR(50) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fetching turns by conversation
CREATE INDEX IF NOT EXISTS idx_turns_conversation_id ON conversation_turns(conversation_id);
CREATE INDEX IF NOT EXISTS idx_turns_order ON conversation_turns(conversation_id, turn_index);

-- =============================================================================
-- 3. Feedback Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS feedback (
    id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    conversation_id VARCHAR(255),
    timestamp TIMESTAMPTZ NOT NULL,
    feedback_type VARCHAR(50) NOT NULL, -- 'good', 'bad', 'corrected'
    quality_score DECIMAL(3, 2), -- 1.00 to 5.00
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    corrected_response TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_feedback_session_id ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_quality ON feedback(quality_score) WHERE quality_score IS NOT NULL;

-- =============================================================================
-- 4. Corrections Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS corrections (
    id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    original_prompt TEXT NOT NULL,
    original_response TEXT NOT NULL,
    corrected_response TEXT NOT NULL,
    correction_type VARCHAR(50) NOT NULL, -- 'factual', 'style', 'format', 'completeness', 'other'
    notes TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_corrections_session_id ON corrections(session_id);
CREATE INDEX IF NOT EXISTS idx_corrections_type ON corrections(correction_type);
CREATE INDEX IF NOT EXISTS idx_corrections_timestamp ON corrections(timestamp DESC);

-- =============================================================================
-- 5. Quality Scores Table (cache)
-- =============================================================================
CREATE TABLE IF NOT EXISTS quality_scores (
    conversation_id VARCHAR(255) PRIMARY KEY,
    overall_score DECIMAL(3, 2) NOT NULL,
    length_score DECIMAL(3, 2) NOT NULL,
    coherence_score DECIMAL(3, 2) NOT NULL,
    feedback_score DECIMAL(3, 2) NOT NULL,
    recency_score DECIMAL(3, 2) NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for filtering by score
CREATE INDEX IF NOT EXISTS idx_quality_overall ON quality_scores(overall_score);

-- =============================================================================
-- 6. Model Versions Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_versions (
    version VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    base_model VARCHAR(255) NOT NULL,
    dataset_size INTEGER DEFAULT 0,
    status VARCHAR(50) NOT NULL DEFAULT 'ready', -- 'ready', 'active', 'failed'
    training_config JSONB DEFAULT '{}',
    adapter_path VARCHAR(500), -- HuggingFace Hub path or local path
    hf_commit_hash VARCHAR(100), -- HuggingFace commit hash
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for status filtering
CREATE INDEX IF NOT EXISTS idx_model_status ON model_versions(status);
CREATE INDEX IF NOT EXISTS idx_model_created ON model_versions(created_at DESC);

-- =============================================================================
-- 7. Active Version Table (single row)
-- =============================================================================
CREATE TABLE IF NOT EXISTS active_model (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1), -- Ensures only one row
    version VARCHAR(255) REFERENCES model_versions(version),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default row if not exists
INSERT INTO active_model (id, version) VALUES (1, NULL)
ON CONFLICT (id) DO NOTHING;

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
DROP TRIGGER IF EXISTS update_conversations_updated_at ON conversations;
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_feedback_updated_at ON feedback;
CREATE TRIGGER update_feedback_updated_at
    BEFORE UPDATE ON feedback
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_corrections_updated_at ON corrections;
CREATE TRIGGER update_corrections_updated_at
    BEFORE UPDATE ON corrections
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_model_versions_updated_at ON model_versions;
CREATE TRIGGER update_model_versions_updated_at
    BEFORE UPDATE ON model_versions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_active_model_updated_at ON active_model;
CREATE TRIGGER update_active_model_updated_at
    BEFORE UPDATE ON active_model
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Row Level Security (RLS) - Optional
-- =============================================================================
-- Uncomment below to enable RLS for additional security

-- ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE conversation_turns ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE corrections ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE quality_scores ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE model_versions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE active_model ENABLE ROW LEVEL SECURITY;

-- Create policies for service role (full access)
-- CREATE POLICY "Service role has full access" ON conversations FOR ALL USING (true);
-- ... (repeat for other tables)

-- =============================================================================
-- Verification
-- =============================================================================
-- Run this to verify all tables were created:
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
